from run_mo_regression import *
from plotting import *

if __name__ == '__main__':
    output_folder = "output_files/mo_regression"
    os.makedirs(output_folder, exist_ok=True)
    if torch.cuda.is_available():
        target_device = "cuda:0"
    else:
        target_device = "cpu"

    cfg = {}
    cfg["mo_mode"] = "per_sample"

    # ---- hyperparameters from tuning experiment ----
    cfg["lr"] = 1e-2
    cfg["betas"] = (0.99, 0.999)

    # ---- other hyperparameters, specific to problem ----
    cfg["n_learning_iterations"] = 20000
    cfg["ref_point"] = (20, 20, 20)
    cfg["n_samples"] = 400
    cfg["train_ratio"] = 0.5
    cfg["n_mo_obj"] = 3
    cfg["n_mo_sol"] = 5
    cfg["loss_names"] = ["MSELoss", "MSELoss", "MSELoss"]

    # ---- training with dynamic loss minimization ----
    mo_optimizer_name = "hv_maximization"
    output_dict = run(mo_optimizer_name, target_device, cfg)

    # ---- plot output ----
    plot_training_process(cfg, output_dict, output_folder)
    plot_mo_regression_predictions(cfg, output_dict, output_folder)
    plot_os_pareto_fronts(cfg, output_dict, output_folder)