#!/usr/bin/env python3
"""
Simple test to verify the new flat structure works.
Tests that model loads and environment runs.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from robot import Robot, Nozzle
from salp_robot_env import SalpRobotEnv

def test_imports():
    """Test that all imports work."""
    print("✓ All imports successful")
    return True

def test_environment():
    """Test that environment can be created."""
    try:
        nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
        robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                      max_contraction=0.06, nozzle=nozzle)
        robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
        robot.set_environment(density=1000)
        
        env = SalpRobotEnv(render_mode=None, robot=robot)
        obs, _ = env.reset()
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False

def test_model_load():
    """Test that model can be loaded from new location."""
    model_path = "../data/models/v1/salp_robot_finalv2.zip"
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found at: {model_path}")
        return False
    
    try:
        nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
        robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                      max_contraction=0.06, nozzle=nozzle)
        robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
        robot.set_environment(density=1000)
        env = SalpRobotEnv(render_mode=None, robot=robot)
        
        model = SAC.load(model_path, env=env, device="cpu")
        print(f"✓ Model loaded successfully from: {model_path}")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"✓ Model prediction works")
        print(f"  Action shape: {action.shape}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("TESTING NEW FLAT STRUCTURE")
    print("="*70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Environment Creation", test_environment),
        ("Model Loading", test_model_load),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        result = test_func()
        results.append(result)
        print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - New structure works!")
    else:
        print("❌ SOME TESTS FAILED - Need to fix issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
