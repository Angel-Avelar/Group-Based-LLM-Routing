#!/bin/bash
set -e  # stop on first error

echo "=== create_users.py ==="
python create_users.py

echo "=== train_em.py 2 ==="
python train_em.py 2

echo "=== test_accuracy.py 2 ==="
python test_accuracy.py 2

echo "=== learned_groups_breakdown.py 2 ==="
python learned_groups_breakdown.py 2

echo "=== entering routing/ ==="
cd routing

echo "=== preference_predict.py ==="
python preference_predict.py

echo "=== results_summary_tables.py ==="
python results_summary_tables.py

echo "=== routing.py ==="
python routing.py

echo "=== routing_summary_tables.py ==="
python routing_summary_tables.py

echo "=== routing_plot.py ==="
python routing_plot.py

echo "=== All done ==="