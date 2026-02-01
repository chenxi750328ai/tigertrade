import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from datetime import datetime
import threading
import time
import glob
import argparse
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class RLTradingNetwork(nn.Module):
    """用于交易决策的强化学习网络"""
    def __init__(self, input_size=12, action_size=3, hidden_size=512, num_layers=4):
        """
        输入: 技术指标和市场状态
        输出: 三种操作的概率分布 (买入, 卖出, 持有)
        """
        super(RLTradingNetwork, self).__init__()
        
        # 输入层
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 多层LSTM用于序列建模
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Q值网络 - 估计每个动作的价值
        self.q_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//4),
            nn.Linear(hidden_size//4, action_size)
        )
        
        # 价值网络 - 估计当前状态的价值
        self.value_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size//4),
            nn.Linear(hidden_size//4, 1)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # 输入变换
        x = self.relu(self.input_layer(x))  # (batch, seq, hidden)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # 使用最后一步的输出
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        
        # 计算Q值和状态价值
        q_values = self.q_network(final_hidden)  # (batch, action_size)
        state_value = self.value_network(final_hidden)  # (batch, 1)
        
        # 计算动作概率（使用优势函数）
        advantages = q_values - state_value
        action_probs = torch.softmax(advantages, dim=-1)
        
        return action_probs, q_values


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = map(np.stack, zip(*[self.buffer[i] for i in batch]))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class RLTradingStrategy:
    """基于强化学习的交易策略"""
    def __init__(self, data_dir='/home/cx/trading_data', model_path=None, learning_rate=1e-4):
        # 强制使用GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            raise RuntimeError("CUDA不可用，此策略需要GPU运行")
        
        self.data_dir = data_dir
        
        # 初始化网络
        self.network = RLTradingNetwork().to(self.device)
        self.target_network = RLTradingNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # 复制参数到目标网络
        self.target_network.load_state_dict(self.network.state_dict())
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(capacity=10000)
        
        # 强化学习参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 1000  # 更新目标网络频率
        self.step_count = 0
        
        # 控制训练和推理的标志
        self.should_train = True
        self.model_lock = threading.Lock()
        
        # 记录交易历史和性能
        self.performance_log = {
            'total_reward': 0,
            'total_steps': 0,
            'win_count': 0,
            'loss_count': 0,
            'total_trades': 0
        }
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"RL Trading Network参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.network.load_state_dict(checkpoint['network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                print(f"✅ 从 {model_path} 加载模型成功")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}，使用初始模型")
        
        # 启动训练线程
        self.training_thread = threading.Thread(target=self.train_continuously, daemon=True)
        self.training_thread.start()
    
    def prepare_features(self, row):
        """从数据行中准备特征向量"""
        try:
            features = [
                row['price_current'],
                row['atr'],
                row['rsi_1m'] if pd.notna(row['rsi_1m']) else 50,
                row['rsi_5m'] if pd.notna(row['rsi_5m']) else 50,
                row.get('boll_upper', 0),
                row.get('boll_mid', 0),
                row.get('boll_lower', 0),
                row.get('boll_position', 0.5),
                row.get('price_change_1', 0),
                row.get('price_change_5', 0),
                row.get('volatility', 0),
                row.get('volume_1m', 0)
            ]
            # 归一化特征
            features_np = np.array(features)
            mean_val = np.mean(features_np)
            std_val = np.std(features_np) + 1e-8
            normalized_features = (features_np - mean_val) / std_val
            return normalized_features.tolist()
        except Exception as e:
            print(f"prepare_features错误: {e}")
            # 返回默认特征值
            return [0.0] * 12
    
    def predict_action(self, current_data):
        """使用模型预测交易动作"""
        with self.model_lock:
            try:
                # 准备输入数据
                features = self.prepare_features(current_data)
                
                # 以一定概率随机探索
                if np.random.random() <= self.epsilon:
                    action = np.random.choice([0, 1, 2])  # 随机选择动作
                    confidence = 0.33  # 随机动作的置信度较低
                    return int(action), float(confidence)
                
                # 使用模型预测
                input_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)  # (1, 1, 10)
                
                # 模型预测
                with torch.no_grad():
                    self.network.eval()
                    action_probs, _ = self.network(input_tensor)
                    action_probs = action_probs.cpu().numpy()[0]
                    
                    # 根据概率选择动作
                    action = np.random.choice(len(action_probs), p=action_probs)
                    confidence = action_probs[action]
                
                return int(action), float(confidence)
            except Exception as e:
                print(f"预测错误: {e}")
                import traceback
                traceback.print_exc()
                return 0, 0.0
    
    def compute_reward(self, action, current_data, prev_data=None):
        """计算强化学习奖励函数"""
        # 基础奖励计算
        # action: 0=持有, 1=买入, 2=卖出
        # reward应该反映交易的盈利能力和风险控制
        
        price_current = current_data['price_current']
        if prev_data is not None:
            prev_price = prev_data['price_current']
            if prev_price > 0:
                price_return = (price_current - prev_price) / prev_price  # 计算收益率
            else:
                return 0.0
        else:
            return 0.0  # 没有前一个状态时返回0奖励
        
        # 根据动作和收益率计算奖励
        if action == 1:  # 买入
            # 买入后价格上涨获得正奖励，下跌获得负奖励
            reward = price_return
        elif action == 2:  # 卖出
            # 卖出后价格下跌获得正奖励，上涨获得负奖励
            reward = -price_return
        else:  # 持有
            # 持有时根据市场趋势获得较小奖励
            reward = abs(price_return) * 0.1
        
        # 放大奖励信号以便更好地训练
        reward *= 100
        
        # 添加一些基于技术指标的奖励修正
        if current_data['rsi_1m'] is not None and current_data['rsi_5m'] is not None:
            # 如果RSI显示超买超卖，且采取了相应的反向操作，给予额外奖励
            if action == 2 and current_data['rsi_1m'] > 70:  # 卖出且超买
                reward += 0.5
            elif action == 1 and current_data['rsi_1m'] < 30:  # 买入且超卖
                reward += 0.5
        
        return float(reward)
    
    def remember(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        """从经验回放缓冲区中采样并训练"""
        if len(self.memory) < batch_size:
            return
        
        # 从经验池中采样
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).unsqueeze(1).to(self.device)  # (batch, 1, input_size)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values, _ = self.network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        next_q_values, _ = self.target_network(next_states)
        next_q_values = next_q_values.max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪以稳定训练
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load_training_data(self):
        """加载训练数据"""
        # 获取最新的数据文件
        all_data_files = []
        
        # 获取所有数据目录
        all_data_dirs = glob.glob(os.path.join(self.data_dir, '202*-*-*'))
        if all_data_dirs:
            # 按日期排序，获取最新的几个文件
            sorted_dirs = sorted(all_data_dirs, reverse=True)
            for data_dir in sorted_dirs[:7]:  # 使用最近7天的数据
                # 包含原始数据和扩展数据
                data_files = glob.glob(os.path.join(data_dir, 'trading_data_*.csv'))
                data_files.extend(glob.glob(os.path.join(data_dir, 'extended_trading_data_*.csv')))
                data_files.extend(glob.glob(os.path.join(data_dir, 'prepared_features_*.csv')))  # 也包含准备好的特征数据
                all_data_files.extend(data_files)
        
        if not all_data_files:
            return None
        
        # 按修改时间排序，获取最新的文件
        data_file = sorted(all_data_files, key=os.path.getmtime, reverse=True)[0]
        
        try:
            df = pd.read_csv(data_file)
            # 清理数据
            df = df.dropna(subset=['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m'])
            
            if len(df) < 10:  # 需要至少10个样本进行训练
                return None
                
            return df
        except Exception as e:
            print(f"加载训练数据错误: {e}")
            return None
    
    def train_model(self, df):
        """使用强化学习训练模型"""
        try:
            # 准备训练数据，使用基于盈利的标签
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            look_ahead = 10  # 向前看10个时间步长来计算盈利
            
            for i in range(1, len(df) - look_ahead):
                prev_row = df.iloc[i-1]
                curr_row = df.iloc[i]
                
                # 计算未来look_ahead步的盈利来确定最佳动作
                current_price = curr_row['price_current']
                future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
                
                if len(future_prices) == 0:
                    continue
                
                # 计算最大盈利和最大亏损
                max_future_price = max(future_prices)
                min_future_price = min(future_prices)
                
                buy_profit = (max_future_price - current_price) / current_price
                sell_profit = (current_price - min_future_price) / current_price
                
                # 确定最佳动作
                profit_threshold = 0.002  # 0.2%的阈值
                if buy_profit > profit_threshold and buy_profit > sell_profit:
                    optimal_action = 1  # 买入
                elif sell_profit > profit_threshold and sell_profit > buy_profit:
                    optimal_action = 2  # 卖出
                else:
                    optimal_action = 0  # 不操作
                
                # 准备状态表示
                prev_state = self.prepare_features(prev_row)
                curr_state = self.prepare_features(curr_row)
                
                # 使用最优动作计算奖励
                reward = self.compute_reward(optimal_action, curr_row, prev_row)
                
                # 存储到经验池
                self.remember(prev_state, optimal_action, reward, curr_state, False)
            
            # 从经验池中训练
            if len(self.memory) >= 32:
                for epoch in range(5):  # 训练5次
                    batch_losses = []
                    for _ in range(10):  # 每次训练抽10个批次
                        if len(self.memory) >= 32:
                            try:
                                # 从经验池中抽样
                                states, actions, rewards, next_states, dones = self.memory.sample(32)
                                
                                # 转换为tensor
                                states = torch.FloatTensor(states).unsqueeze(1).to(self.device)  # (batch, 1, input_size)
                                actions = torch.LongTensor(actions).to(self.device)
                                rewards = torch.FloatTensor(rewards).to(self.device)
                                next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
                                dones = torch.BoolTensor(dones).to(self.device)
                                
                                # 计算当前Q值
                                current_q_values, _ = self.network(states)
                                current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
                                
                                # 计算目标Q值
                                next_q_values, _ = self.target_network(next_states)
                                next_q_values = next_q_values.max(1)[0].detach()
                                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
                                
                                # 计算损失
                                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                                
                                # 反向传播
                                self.optimizer.zero_grad()
                                loss.backward()
                                # 梯度裁剪以稳定训练
                                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                                self.optimizer.step()
                                
                                batch_losses.append(loss.item())
                            except:
                                continue
                    
                    if batch_losses:
                        avg_loss = sum(batch_losses) / len(batch_losses)
                        print(f"RL训练轮次 {epoch+1}/5, 平均损失: {avg_loss:.4f}")
            
            # 更新目标网络
            self.step_count += 1
            if self.step_count % self.update_target_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
                print(f"✅ 目标网络已更新 (step: {self.step_count})")
        
        except Exception as e:
            print(f"RL模型训练过程错误: {e}")
            import traceback
            traceback.print_exc()
    
    def train_continuously(self):
        """连续训练模型的后台线程"""
        while self.should_train:
            try:
                # 加载数据
                df = self.load_training_data()
                if df is not None and len(df) > 0:
                    print(f"开始RL训练，数据量: {len(df)}")
                    with self.model_lock:
                        self.train_model(df)
                    print("RL模型训练完成")
                
                # 训练较慢，每30分钟训练一次
                time.sleep(1800)
                
            except Exception as e:
                print(f"RL训练线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # 出错后等待1分钟后继续
    
    def save_model(self, path):
        """保存模型"""
        with self.model_lock:
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, path)
            print(f"模型已保存到 {path}")
    
    def log_performance(self, actual_action, predicted_action, reward):
        """记录模型性能"""
        with self.model_lock:
            self.performance_log['total_reward'] += reward
            self.performance_log['total_steps'] += 1
            
            if reward > 0:
                self.performance_log['win_count'] += 1
            elif reward < 0:
                self.performance_log['loss_count'] += 1
                
            self.performance_log['total_trades'] += 1
            
            # 打印性能摘要
            if self.performance_log['total_steps'] > 0:
                avg_reward = self.performance_log['total_reward'] / self.performance_log['total_steps']
                win_rate = self.performance_log['win_count'] / max(self.performance_log['total_trades'], 1)
                print(f"📊 RL模型性能 - 平均奖励: {avg_reward:.3f}, 胜率: {win_rate:.3f}, ε: {self.epsilon:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning Trading Strategy')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict',
                        help='运行模式: train(仅训练), predict(仅预测)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存或加载路径')
    args = parser.parse_args()
    
    strategy = RLTradingStrategy(model_path=args.model_path)
    
    if args.mode == 'train':
        print("仅运行RL训练模式...")
        # 训练模式，不退出
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("RL训练已停止")
    elif args.mode == 'predict':
        print("运行RL预测模式...")
        # 演示如何使用
        sample_data = {
            'price_current': 93.5,
            'grid_lower': 93.0,
            'grid_upper': 94.0,
            'atr': 0.2,
            'rsi_1m': 30.0,
            'rsi_5m': 40.0,
            'buffer': 0.05,
            'threshold': 93.05,
            'near_lower': True,
            'rsi_ok': True
        }
        
        action, confidence = strategy.predict_action(sample_data)
        action_map = {0: "持有", 1: "买入", 2: "卖出"}
        
        print(f"🧠 RL模型预测: {action_map[action]}, 置信度: {confidence:.3f}")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("程序已停止")
    
    # 如果提供了保存路径，保存模型
    if args.model_path:
        strategy.save_model(args.model_path)


if __name__ == "__main__":
    main()