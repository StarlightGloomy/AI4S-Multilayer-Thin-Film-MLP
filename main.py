import os

def main():
    print("Step 1: Generate dataset")
    os.system('python generate_data.py')
    print("Step 2: Train MLP")
    os.system('python train.py')
    print("Step 3: Train size experiment")
    os.system('python train_size_experiment.py')
    print("Step 4: Screen candidates")
    os.system('python screen_design.py')
    print("Step 5: Verify designs")
    os.system('python verify_design.py')
    print("Step 6: Plot results")
    os.system('python plot_results.py')
    print("All done.")

if __name__ == '__main__':
    main()