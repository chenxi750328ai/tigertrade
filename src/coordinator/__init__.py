"""
多Agent协调器
解决Agent之间的同步、互斥、通信问题
"""

from .coordinator import AgentCoordinator
from .agent_wrapper import CoordinatedAgent

__all__ = ['AgentCoordinator', 'CoordinatedAgent']
