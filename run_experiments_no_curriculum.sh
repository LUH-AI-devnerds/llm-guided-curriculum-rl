# These will all log every 10,000 episodes (100 log entries total)
#Amirreza
# python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type conservative_dynamic --episodes 1000000 --deck-type 1-deck --eval-episodes 100000
# python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type conservative_dynamic --episodes 1000000 --deck-type 4-deck --eval-episodes 100000
# python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type conservative_dynamic --episodes 1000000 --deck-type 8-deck --eval-episodes 100000
# python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type conservative_dynamic --episodes 1000000 --deck-type inifnite --eval-episodes 100000
# python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type win_focused --episodes 1000000 --deck-type 1-deck --eval-episodes 100000
# python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type win_focused --episodes 1000000 --deck-type 4-deck --eval-episodes 100000
#Efe
python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type win_focused --episodes 1000000 --deck-type 8-deck --eval-episodes 100000 --budget 10000000
python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type win_focused --episodes 1000000 --deck-type inifnite --eval-episodes 100000 --budget 10000000
python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type balanced --episodes 1000000 --deck-type 1-deck --eval-episodes 100000 --budget 10000000
python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type balanced --episodes 1000000 --deck-type 4-deck --eval-episodes 100000 --budget 10000000
python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type balanced --episodes 1000000 --deck-type 8-deck --eval-episodes 100000 --budget 10000000
python scripts/curriculum_multi_agent_rl.py --no-curriculum --reward-type balanced --episodes 1000000 --deck-type inifnite --eval-episodes 100000 --budget 10000000