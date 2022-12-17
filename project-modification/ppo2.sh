#!/bin/bash
nohup python runPPOmiao2.py -folder logs-PPO2-xi0.1 --confidence_xi 0.1 &

nohup python runPPOmiao2.py -folder logs-PPO2-xi0.2 --confidence_xi 0.2 &

nohup python runPPOmiao2.py -folder logs-PPO2-xi0.5 --confidence_xi 0.5 &

nohup python runPPOmiao2.py -folder logs-PPO2-xi0.95 --confidence_xi 0.95 &

nohup python runPPOmiao2.py -folder logs-PPO2-xi2.0 --confidence_xi 2.0 &

nohup python runPPOmiao2.py -folder logs-PPO2-xi3.0 --confidence_xi 3.0 &

nohup python runPPOmiao2.py -folder logs-PPO2-xi4.0 --confidence_xi 4.0 &