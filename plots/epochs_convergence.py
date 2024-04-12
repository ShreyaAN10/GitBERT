import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description="Script to plot convergence plots")
parser.add_argument("-task", choices=["ns", "jc"], help="ns for Next-Sentence, jc for Job-Completion")
parser.add_argument("-mode", choices=["acc", "loss"], help="ns for Next-Sentence, jc for Job-Completion")
args = parser.parse_args()

if args.task == 'ns':
    # NS BASE
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_base_uncased_constant_10e.pkl", "rb") as f:
        base_constant_ns = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_base_uncased_inverse_sqrt_10e.pkl", "rb") as f:
        base_inv_sqrt_ns = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_base_uncased_polynomial_10e.pkl", "rb") as f:
        base_poly_ns = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_base_uncased_slanted_triangular_10e.pkl", "rb") as f:
        base_sla_tri_ns = pickle.load(f)
    # NS PRETRAINED
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_PRETRAINED_constant_10e.pkl", "rb") as f:
        pre_constant_ns = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_PRETRAINED_inverse_sqrt_10e.pkl", "rb") as f:
        pre_inv_sqrt_ns = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_PRETRAINED_polynomial_10e.pkl", "rb") as f:
        pre_poly_ns = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/NS/finetuned_bert_PRETRAINED_slanted_triangular_10e.pkl", "rb") as f:
        pre_sla_tri_ns = pickle.load(f)
elif args.task == 'jc':
    # JC BASE
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_BASE_constant_10e.pkl", "rb") as f:
        base_constant_jc = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_BASE_inverse_sqrt_10e.pkl", "rb") as f:
        base_inv_sqrt_jc = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_BASE_polynomial_10e.pkl", "rb") as f:
        base_poly_jc = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_BASE_slanted_triangular_10e.pkl", "rb") as f:
        base_sla_tri_jc = pickle.load(f)
    # JC PRETRAINED
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_PRETRAINED_constant_10e.pkl", "rb") as f:
        pre_constant_jc = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_PRETRAINED_inverse_sqrt_10e.pkl", "rb") as f:
        pre_inv_sqrt_jc = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_PRETRAINED_polynomial_10e.pkl", "rb") as f:
        pre_poly_jc = pickle.load(f)
    with open("/home/s4an/bert-autocom/models/JC/JC_finetuned_bert_PRETRAINED_slanted_triangular_10e.pkl", "rb") as f:
        pre_sla_tri_jc = pickle.load(f)

if args.task == 'ns':
    ns_list = [base_constant_ns, base_inv_sqrt_ns, base_poly_ns, base_sla_tri_ns, pre_constant_ns, pre_inv_sqrt_ns, pre_poly_ns, pre_sla_tri_ns]
elif args.task == 'jc':
    jc_list = [base_constant_jc, base_inv_sqrt_jc, base_poly_jc, base_sla_tri_jc, pre_constant_jc, pre_inv_sqrt_jc, pre_poly_jc, pre_sla_tri_jc]


if args.task == 'ns' and args.mode == 'loss':
    # Next-Sentence Convergence
    plt.plot(ns_list[0]['epoch'], ns_list[0]['train_loss'], label="Base-Constant")
    plt.plot(ns_list[1]['epoch'], ns_list[1]['train_loss'], label="Base-Inverse-SquareRoot")
    plt.plot(ns_list[2]['epoch'], ns_list[2]['train_loss'], label="Base-Polynomial")
    plt.plot(ns_list[3]['epoch'], ns_list[3]['train_loss'], label="Base-Slanted-Triangular")
    plt.plot(ns_list[4]['epoch'], ns_list[4]['train_loss'], label="Pretrained-Constant")
    plt.plot(ns_list[5]['epoch'], ns_list[5]['train_loss'], label="Pretrained-Inverse-SquareRoot")
    plt.plot(ns_list[6]['epoch'], ns_list[6]['train_loss'], label="Pretrained-Polynomial")
    plt.plot(ns_list[7]['epoch'], ns_list[7]['train_loss'], label="Pretrained-Slanted-Triangular")
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Convergence on the Next-Sentence task')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"NS-loss.png")
if args.mode == 'acc' and args.task == 'ns':
    # Next-Sentence Convergence
    plt.plot(ns_list[0]['epoch'], ns_list[0]['train_accuracy'], label="Base-Constant")
    plt.plot(ns_list[1]['epoch'], ns_list[1]['train_accuracy'], label="Base-Inverse-SquareRoot")
    plt.plot(ns_list[2]['epoch'], ns_list[2]['train_accuracy'], label="Base-Polynomial")
    plt.plot(ns_list[3]['epoch'], ns_list[3]['train_accuracy'], label="Base-Slanted-Triangular")
    plt.plot(ns_list[4]['epoch'], ns_list[4]['train_accuracy'], label="Pretrained-Constant")
    plt.plot(ns_list[5]['epoch'], ns_list[5]['train_accuracy'], label="Pretrained-Inverse-SquareRoot")
    plt.plot(ns_list[6]['epoch'], ns_list[6]['train_accuracy'], label="Pretrained-Polynomial")
    plt.plot(ns_list[7]['epoch'], ns_list[7]['train_accuracy'], label="Pretrained-Slanted-Triangular")
    plt.xlabel('Epochs')
    plt.ylabel('Train Accuracy')
    plt.title('Training Accuracy on the Next-Sentence task')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"NS-train.png")

if args.task == 'ns' and args.mode == 'loss':
    # Job-Completion Convergence
    plt.plot(jc_list[0]['epoch'], jc_list[0]['train_loss'], label="Base-Constant")
    plt.plot(jc_list[1]['epoch'], jc_list[1]['train_loss'], label="Base-Inverse-SquareRoot")
    plt.plot(jc_list[2]['epoch'], jc_list[2]['train_loss'], label="Base-Polynomial")
    plt.plot(jc_list[3]['epoch'], jc_list[3]['train_loss'], label="Base-Slanted-Triangular")
    plt.plot(jc_list[4]['epoch'], jc_list[4]['train_loss'], label="Pretrained-Constant")
    plt.plot(jc_list[5]['epoch'], jc_list[5]['train_loss'], label="Pretrained-Inverse-SquareRoot")
    plt.plot(jc_list[6]['epoch'], jc_list[6]['train_loss'], label="Pretrained-Polynomial")
    plt.plot(jc_list[7]['epoch'], jc_list[7]['train_loss'], label="Pretrained-Slanted-Triangular")
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Convergence on the Job-Completion task')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"JC-loss.png")
elif args.task == 'jc' and args.mode == 'acc':
    # Job-Completion Convergence
    plt.plot(jc_list[0]['epoch'], jc_list[0]['train_accuracy'], label="Base-Constant")
    plt.plot(jc_list[1]['epoch'], jc_list[1]['train_accuracy'], label="Base-Inverse-SquareRoot")
    plt.plot(jc_list[2]['epoch'], jc_list[2]['train_accuracy'], label="Base-Polynomial")
    plt.plot(jc_list[3]['epoch'], jc_list[3]['train_accuracy'], label="Base-Slanted-Triangular")
    plt.plot(jc_list[4]['epoch'], jc_list[4]['train_accuracy'], label="Pretrained-Constant")
    plt.plot(jc_list[5]['epoch'], jc_list[5]['train_accuracy'], label="Pretrained-Inverse-SquareRoot")
    plt.plot(jc_list[6]['epoch'], jc_list[6]['train_accuracy'], label="Pretrained-Polynomial")
    plt.plot(jc_list[7]['epoch'], jc_list[7]['train_accuracy'], label="Pretrained-Slanted-Triangular")
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Training accuracy on the Job-Completion task')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"JC-acc.png")  
 
