.PHONY: test test-integration test-regression ci-check lint

# 运行所有测试
test:
	@echo "🧪 运行所有测试..."
	python -m pytest tests/ -v

# 运行集成测试
test-integration:
	@echo "🔗 运行集成测试..."
	python tests/test_run_moe_demo_integration.py

# 运行回归测试
test-regression:
	@echo "🔄 运行回归测试..."
	@echo "1. 检查下单逻辑..."
	@python -c "import sys; sys.path.insert(0, '.'); \
		content = open('scripts/run_moe_demo.py').read(); \
		checks = {'place_tiger_order': 'place_tiger_order' in content and not all(l.strip().startswith('#') for l in content.split('\n') if 'place_tiger_order' in l), \
		          'check_risk_control': 'check_risk_control' in content, \
		          '执行买入': '执行买入' in content, \
		          '执行卖出': '执行卖出' in content}; \
		failed = [k for k, v in checks.items() if not v]; \
		sys.exit(0 if not failed else (print(f'❌ 失败: {failed}') or 1))"
	@echo "2. 运行集成测试..."
	@python tests/test_run_moe_demo_integration.py
	@echo "3. 运行基础功能测试..."
	@python -m pytest tests/test_place_tiger_order.py -v
	@echo "✅ 回归测试完成"

# CI检查（在CI环境中运行）
ci-check: test-regression
	@echo "✅ CI检查通过"

# 代码检查
lint:
	@echo "🔍 代码检查..."
	@python -m flake8 scripts/run_moe_demo.py --max-line-length=120 --ignore=E501,W503 || true

# 快速检查（不运行完整测试）
quick-check:
	@echo "⚡ 快速检查..."
	@python -c "import sys; sys.path.insert(0, '.'); \
		content = open('scripts/run_moe_demo.py').read(); \
		assert 'place_tiger_order' in content, '缺少下单逻辑'; \
		assert 'check_risk_control' in content, '缺少风控检查'; \
		print('✅ 快速检查通过')"
