import matplotlib.pyplot as plt
import argparse
import pickle
import seaborn as sns  # Import Seaborn for color palettes

def plot_convergence(train_list, eval_list):
    plt.figure(figsize=(12, 6))
    line_styles = ['-', '--', '-.', ':']  # Different line patterns
    labels = []  # List to store scheduler labels
    # Set Seaborn style and color palette
    sns.set_style("whitegrid")
    palette = sns.color_palette()  # Choose a color palette

    # Plot training loss
    plt.subplot(2, 2, 1)
    j = 0
    for results in train_list:
        scheduler = results['scheduler']
        min_loss = [results['loss'][0]]
        for i in range(1, len(results['loss'])):
            if results['loss'][i] < min_loss[-1]:
                min_loss.append(results['loss'][i])  
            else:
                min_loss.append(min_loss[-1])
        # plt.plot(range(1, len(min_loss) + 1), min_loss, label=f'{scheduler} Scheduler', 
        #         linestyle=line_styles[0], color=palette[j])
        plt.plot(range(1, len(results['loss']) + 1), results['loss'], label=f'{scheduler} Scheduler', 
                linestyle=line_styles[0], color=palette[j])
        j += 1
    plt.xlabel('Batches')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Convergence')
    plt.legend(loc='best')

    # Plot training loss
    plt.subplot(2, 2, 2)
    i = 0
    for results in train_list:
        scheduler = results['scheduler']
        # print(results.keys())
        print(results['loss'][:100])
        print(results['accuracy'][:100])
        plt.plot(range(1, 100 + 1), results['loss'][:100], label=f'{scheduler} Scheduler', 
                linestyle=line_styles[0], color=palette[i])
        i += 1
    plt.xlabel('Batches')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Convergence Zoomed In')
    # plt.legend(loc='best')

    # Plot validation accuracy
    plt.subplot(2, 2, 3)
    j = 0
    for results in eval_list:
        scheduler = results['scheduler']
        max_acc = [results['accuracy'][0]]
        for i in range(1, len(results['accuracy'])):
            if results['accuracy'][i] > max_acc[-1]:
                max_acc.append(results['accuracy'][i])  
            else:
                max_acc.append(max_acc[-1])
        plt.plot(range(1, len(max_acc) + 1), max_acc, label=f'{scheduler} Scheduler', 
                linestyle=line_styles[0], color=palette[j])
        j += 1
    plt.xlabel('Batches')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Convergence')
    plt.legend(loc='best')

    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    i = 0
    for results in eval_list:
        scheduler = results['scheduler']
        plt.plot(range(1, 100 + 1), results['accuracy'][:100], label=f'{scheduler} Scheduler', 
                linestyle=line_styles[0], color=palette[i])
        i += 1
    plt.xlabel('Batches')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Convergence')
    # plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
    plt.savefig("convergence.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning with different learning rate schedulers")
    parser.add_argument("-results", type=str, choices=['train', 'test', 'eval'], help="Learning rate scheduler")
    args = parser.parse_args()

    with open("/home/s4an/bert-autocom/models/constant_intermediate_train.pkl", "rb") as f:
        con_1e_train = pickle.load(f)
        con_1e_train['scheduler'] = 'Constant (C-LR)'
    with open("/home/s4an/bert-autocom/models/constant_intermediate_eval.pkl", "rb") as f:
        con_1e_eval = pickle.load(f)
        con_1e_eval['scheduler'] = 'Constant (C-LR)'

    with open("/home/s4an/bert-autocom/models/polynomial_intermediate_train.pkl", "rb") as f:
        ply_1e_train = pickle.load(f)
        ply_1e_train['scheduler'] = 'Polynomial Decay (PD-LR)'
    with open("/home/s4an/bert-autocom/models/polynomial_intermediate_eval.pkl", "rb") as f:
        ply_1e_eval = pickle.load(f)
        ply_1e_eval['scheduler'] = 'Polynomial Decay (PD-LR)'

    with open("/home/s4an/bert-autocom/models/inverse_sqrt_intermediate_train.pkl", "rb") as f:
        inverse_sqrt_1e_train = pickle.load(f)
        inverse_sqrt_1e_train['scheduler'] = 'Inverse Square Root (ISQ-LR)'
    with open("/home/s4an/bert-autocom/models/inverse_sqrt_intermediate_eval.pkl", "rb") as f:
        inverse_sqrt_1e_eval = pickle.load(f)
        inverse_sqrt_1e_eval['scheduler'] = 'Inverse Square Root (ISQ-LR)'

    with open("/home/s4an/bert-autocom/models/slanted_triangular_intermediate_train.pkl", "rb") as f:
        slanted_tri_1e_train = pickle.load(f)
        slanted_tri_1e_train['scheduler'] = 'Slanted Triangular (ST-LR)'
    with open("/home/s4an/bert-autocom/models/slanted_triangular_intermediate_eval.pkl", "rb") as f:
        slanted_tri_1e_eval = pickle.load(f)
        slanted_tri_1e_eval['scheduler'] = 'Slanted Triangular (ST-LR)'

    if args.results == 'train':
        train_list = [con_1e_train, ply_1e_train, inverse_sqrt_1e_train, slanted_tri_1e_train]
        eval_list = [con_1e_eval, ply_1e_eval, inverse_sqrt_1e_eval, slanted_tri_1e_eval]
        plot_convergence(train_list, eval_list)
    elif args.results == 'test':
        print("nothing to do..")
    elif args.results == 'eval':
        eval_list = [con_1e_eval, ply_1e_eval, inverse_sqrt_1e_eval, slanted_tri_1e_eval]
        plot_convergence(eval_list)
    
