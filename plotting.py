import torch
import os
import numpy as np
import pdb
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from mo_optimizers.utils import generate_k_preferences


def plot_training_process(cfg, output_dict, output_folder):
    # plot settings
    node_size = 15
    line_width = 3
    font_size = 22
    legend_font_size = 22
    y_label_list = [r'$\mathrm{MSE}(\cos(x),z)$',r'$\mathrm{MSE}(\sin(x),z)$',r'$\mathrm{MSE}(\sin(x+\pi),z)$']
    linear_threshold = -10**(-2)

    color_list = plt.cm.get_cmap('tab10', 10)
    n_mo_obj = cfg["n_mo_obj"]
    n_mo_sol = cfg["n_mo_sol"]

    hv_array = output_dict["training_hv"] #n_learning_iterations
    training_loss = output_dict["training_loss"]  #n_learning_iterations * n_mo_obj * n_mo_sol
    n_iterations = training_loss.shape[0]

    x_label = 'Iterations'
    y_label = r'$\mathrm{HV}$'
    title = 'Training metrics over iterations'
    image_file_name = '{}d_training_metrics'.format(n_mo_obj)
    xtick_choices = [0, int(n_iterations/4), int(n_iterations/2), int(n_iterations* 3/4), n_iterations]
    xticklabel_choices = [0, None, int(n_iterations/2), None, n_iterations]
    ytick_choices_mse = []
    yticklabel_choices_mse = []
    ytick_choices_mse.append([0, 1])
    ytick_choices_mse.append([0, 1, 2])
    ytick_choices_mse.append([0, 1, 2])


    # ---- stuff for linear and log scale mix ----
    hv_max = np.max(hv_array)
    neg_hv_array_diff = -1 * (hv_max - hv_array)
    largest_log = np.floor(np.log10(hv_max))
    smallest_power = -2
    chosen_ytick_marks = list()
    chosen_ytick_labels = list()
    chosen_ytick_marks.append(0)
    rounded_hv_max = np.round(hv_max,-1 * smallest_power + 1)
    chosen_ytick_labels.append(r'$'+str(rounded_hv_max)+'$')
    power_array = np.arange(smallest_power,(largest_log+1))
    for ten_power in power_array:
        chosen_ytick_marks.append(-1 * 10**ten_power)
        cur_label = hv_max - 10**ten_power
        rounded_cur_label = np.round(cur_label,-1 * smallest_power + 1)
        chosen_ytick_labels.append(r'$' + str(rounded_hv_max) + ' -' + '10^{' + str(int(ten_power)) +  '}' + '$')

    fig,cur_ax = plt.subplots(1 + n_mo_obj,figsize = (10,10), dpi = 100, sharex = 'col')
    cur_ax[0].plot(neg_hv_array_diff,zorder = 0,c = 'black', linewidth = line_width)
    cur_ax[0].set_ylabel(r'$\mathrm{HV}$', fontsize = font_size)
    cur_ax[0].set_yscale('symlog',linthresh = -1*linear_threshold) # -1* to cancel out minus
    cur_ax[0].set_yticks(chosen_ytick_marks)
    cur_ax[0].set_yticklabels(chosen_ytick_labels)
    cur_ax[0].tick_params(axis='y',which = 'both', labelsize = font_size)
    cur_ax[0].set_xticks(xtick_choices)

    # add gray line to indicate change from log to linear scale
    xlim = [-0.01*n_iterations,1.01*n_iterations]
    cur_ax[0].plot([xlim[0],xlim[1]],[linear_threshold, linear_threshold], c='gray', linestyle='--', linewidth=1, zorder=-1)
    if image_file_name[0:2] == '2d':
        linear_label_space = 0.001
        log_label_space = 0.02
    elif image_file_name[0:2] == '3d':
        linear_label_space = 0.001
        log_label_space = 0.04
    else:
        raise ValueError('Unknown image prefix.')
    cur_ax[0].text(n_iterations, linear_threshold + linear_label_space, r'linear $\uparrow$', c='gray',\
                         horizontalalignment='right', fontsize=font_size-6)
    cur_ax[0].text(n_iterations, linear_threshold - log_label_space, r'logarithmic $\downarrow$',c = 'gray',\
                         horizontalalignment='right', fontsize=font_size-6)
    cur_ax[0].set_xlim(xlim[0],xlim[1])

    for i_mo_obj in range(0,n_mo_obj):
        legend_line_list = list()
        legend_label_list = list()

        for i_mo_sol in range(0,n_mo_sol):
            cur_loss = training_loss[:, i_mo_obj, i_mo_sol]
            line_handle, = cur_ax[1 + i_mo_obj].plot(cur_loss, zorder=0, color=color_list(i_mo_sol), linewidth=line_width)
            legend_line_list.append(line_handle)
            legend_label_list.append('Net ' + str(i_mo_sol+1))

        cur_ax[1 + i_mo_obj].set_ylabel(y_label_list[i_mo_obj], fontsize=font_size-4)
        cur_ax[1 + i_mo_obj].set_yticks(ytick_choices_mse[i_mo_obj]) # dont index ytick_choices_mse with 1 + because this list is only for mse
        cur_ax[1 + i_mo_obj].tick_params(axis='y', which='both', labelsize=font_size)
        cur_ax[1 + i_mo_obj].set_xticks(xtick_choices)


    cur_ax[-1].set_xlabel(x_label, fontsize=font_size)
    cur_ax[-1].set_xticks(xtick_choices)
    cur_ax[-1].set_xticklabels(xticklabel_choices, fontsize=font_size)
    plt.savefig(os.path.join(output_folder, image_file_name +'.png'), bbox_inches="tight")
    plt.close(fig)


def plot_mo_regression_predictions_2D(cfg, output_dict, output_folder):
    """
    plotting mo regression predictions
    """
    # plot settings
    color_list = plt.cm.get_cmap('tab10',10)
    node_size = 15
    line_width = 5
    font_size = 22
    legend_font_size = 22
    selected_samples= np.arange(5, 100, 20) # 0.25pi to 0.55pi

    n_losses = cfg["n_mo_obj"]
    n_mo_sol = cfg["n_mo_sol"]
    inputs = output_dict["validation_data_x"]
    targets = output_dict["validation_data_y"]
    validation_output = output_dict["validation_output"]

    # sort inputs in increasing order
    sort_indices = np.argsort(inputs)
    inputs, targets = inputs[sort_indices], targets[sort_indices]
    validation_output = validation_output[:, sort_indices]

    # sort solutions in according to increaing loss 1, to match colors with pareto front
    mo_obj_val_per_sample = output_dict["validation_loss"] #n_samples * n_mo_obj * n_mo_sol
    sort_indices_sol = np.argsort(mo_obj_val_per_sample.mean(axis=0)[0, :])
    validation_output = validation_output[sort_indices_sol, :]
    
    figHandle, axHandle = plt.subplots(1, 1, figsize=(10,5))
    line_handle_ground_truth_zero, = axHandle.plot(inputs, targets[:, 0], color='darkgray', linewidth=line_width, zorder=1)
    line_handle_ground_truth_one, = axHandle.plot(inputs, targets[:, 1], color='black', linewidth=line_width, zorder=1)
    if n_losses == 3:
        line_handle_ground_truth_two, = axHandle.plot(inputs, targets[:, 2], color='gray', linewidth=line_width, zorder=1)
    axHandle.set_xlabel('network input (x)', fontsize=font_size)
    axHandle.set_ylabel('network output (z)', fontsize=font_size)

    line_handle_subplot_zero = list()
    legend_entry_list = list()
    line_handle_subplot_zero.append(line_handle_ground_truth_zero)
    line_handle_subplot_zero.append(line_handle_ground_truth_one)
    if n_losses == 3:
        line_handle_subplot_zero.append(line_handle_ground_truth_two)
    legend_entry_list.append('cos(x)')
    legend_entry_list.append('sin(x)')
    if n_losses == 3:
        legend_entry_list.append(r'sin(x+$\pi$)')
    for i_mo_sol in range(0, n_mo_sol):
        _ = axHandle.scatter(inputs, validation_output[i_mo_sol, :], color=color_list(i_mo_sol), s=node_size, zorder=2, alpha=1/2)
        line_handle = axHandle.scatter(inputs[selected_samples], validation_output[i_mo_sol, selected_samples], \
            color=color_list(i_mo_sol), s=6*node_size, zorder=2, alpha=1.00)
        line_handle_subplot_zero.append(line_handle)
        legend_entry_list.append('Net ' + str(i_mo_sol+1))
    
    axHandle.legend(line_handle_subplot_zero, legend_entry_list, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=legend_font_size) 
    axHandle.set_xticks(np.array([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, 1*np.pi, 1.5*np.pi, 2*np.pi]))
    axHandle.set_xticklabels([r'$0$',r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', \
                        r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$'], fontsize=font_size)
    axHandle.set_yticks(np.array([-1, -0.5, 0, 0.5, 1]))
    axHandle.set_yticklabels([r'$-1$', r'$-\frac{1}{2}$', r'$0$', r'$\frac{1}{2}$', r'$1$', ], fontsize=font_size)
    axHandle.grid(b=True, which='major', axis='both', linestyle='-')

    plt.savefig(os.path.join(output_folder, "{}d_predictions.png".format(n_losses)), bbox_inches="tight")
    plt.close(figHandle)


def plot_mo_regression_predictions_3D(cfg, output_dict, output_folder):
    """
    plotting mo regression predictions
    """
    # plot settings
    color_list = plt.cm.get_cmap('tab10',10)
    node_size = 15
    line_width = 5
    font_size = 22
    legend_font_size = 22
    selected_samples= np.arange(5, 100, 20) # 0.25pi to 0.55pi

    n_losses = cfg["n_mo_obj"]
    n_mo_sol = cfg["n_mo_sol"]
    inputs = output_dict["validation_data_x"]
    targets = output_dict["validation_data_y"]
    validation_output = output_dict["validation_output"]

    # sort inputs in increasing order
    sort_indices = np.argsort(inputs)
    inputs, targets = inputs[sort_indices], targets[sort_indices]
    validation_output = validation_output[:, sort_indices]

    figHandle, axHandle = plt.subplots(1, 1, figsize=(10,5))
    line_handle_ground_truth_zero, = axHandle.plot(inputs, targets[:, 0], color='darkgray', linewidth=line_width, zorder=1)
    line_handle_ground_truth_one, = axHandle.plot(inputs, targets[:, 1], color='black', linewidth=line_width, zorder=1)
    if n_losses == 3:
        line_handle_ground_truth_two, = axHandle.plot(inputs, targets[:, 2], color='gray', linewidth=line_width, zorder=1)
    axHandle.set_xlabel('network input (x)', fontsize=font_size)
    axHandle.set_ylabel('network output (z)', fontsize=font_size)

    line_handle_subplot_zero = list()
    legend_entry_list = list()
    line_handle_subplot_zero.append(line_handle_ground_truth_zero)
    line_handle_subplot_zero.append(line_handle_ground_truth_one)
    if n_losses == 3:
        line_handle_subplot_zero.append(line_handle_ground_truth_two)
    legend_entry_list.append('cos(x)')
    legend_entry_list.append('sin(x)')
    if n_losses == 3:
        legend_entry_list.append(r'sin(x+$\pi$)')
    for i_mo_sol in range(0, n_mo_sol):
        _ = axHandle.scatter(inputs, validation_output[i_mo_sol, :], color=color_list(i_mo_sol), s=node_size, zorder=2, alpha=1/2)
        line_handle = axHandle.scatter(inputs[selected_samples], validation_output[i_mo_sol, selected_samples], \
            color=color_list(i_mo_sol), s=6*node_size, zorder=2, alpha=1.00)
        line_handle_subplot_zero.append(line_handle)
        legend_entry_list.append('Net ' + str(i_mo_sol+1))
    
    axHandle.legend(line_handle_subplot_zero, legend_entry_list, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=legend_font_size) 
    axHandle.set_xticks(np.array([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, 1*np.pi, 1.5*np.pi, 2*np.pi]))
    axHandle.set_xticklabels([r'$0$',r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$', \
                        r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$'], fontsize=font_size)
    axHandle.set_yticks(np.array([-1, -0.5, 0, 0.5, 1]))
    axHandle.set_yticklabels([r'$-1$', r'$-\frac{1}{2}$', r'$0$', r'$\frac{1}{2}$', r'$1$', ], fontsize=font_size)
    axHandle.grid(b=True, which='major', axis='both', linestyle='-')

    plt.savefig(os.path.join(output_folder, "{}d_predictions.png".format(n_losses)), bbox_inches="tight")
    plt.close(figHandle)


def plot_mo_regression_predictions(cfg, output_dict, output_folder):
    """
    plot the NNs' positions in objective space
    """
    mo_obj_val_per_sample = output_dict["validation_loss"] #n_samples * n_mo_obj * n_mo_sol
    n_samples, n_mo_obj, n_mo_sol = mo_obj_val_per_sample.shape

    if n_mo_obj==2:
        plot_mo_regression_predictions_2D(cfg, output_dict, output_folder)
    elif n_mo_obj==3:
        plot_mo_regression_predictions_3D(cfg, output_dict, output_folder)
    else:
        raise NotImplementedError("plotting implemented only for 2 or 3 objectives.")


def plot_os_pareto_fronts_2D(cfg, output_dict, output_folder):
    """
    plot the NNs' positions in objective space
    """
    color_list = plt.cm.get_cmap('tab10',10)
    node_size = 15
    line_width = 5
    font_size = 22
    legend_font_size = 22

    ytick_choices = []
    yticklabel_choices = []
    ytick_choices.append([0, 0.5, 1, 1.5, 2])
    ytick_choices.append([0, 0.5, 1, 1.5, 2])
    label_list = [r'$(\cos(x)-z)^{2}$',r'$(\sin(x)-z)^{2}$',r'$(\sin(x+\pi)-z)^{2}$']
    
    inputs = output_dict["validation_data_x"]
    targets = output_dict["validation_data_y"]
    mo_obj_val_per_sample = output_dict["validation_loss"] #n_samples * n_mo_obj * n_mo_sol
    n_samples, n_mo_obj, n_mo_sol = mo_obj_val_per_sample.shape

    # sort inputs in increasing order
    sort_indices = np.argsort(inputs)
    inputs = inputs[sort_indices]
    mo_obj_val_per_sample = mo_obj_val_per_sample[sort_indices, :, :]
    selected_samples= np.arange(5, 100, 20) # 0.25pi to 0.55pi
    grayscale_list = np.linspace(0, 0.75, len(selected_samples))

    # sort solutions in according to trade-off (loss 0/loss 1), so that the plotted lines do not zigzag
    for i_sample in range(0,n_samples):
        sort_indices_sol = np.argsort(mo_obj_val_per_sample[i_sample,0,:]/mo_obj_val_per_sample[i_sample,1,:])
        mo_obj_val_per_sample[i_sample, :, :] = mo_obj_val_per_sample[i_sample, :, sort_indices_sol].T # transpose because sorting changes dimensionality

    fig, cur_ax = plt.subplots(figsize=(10,10), dpi=100)
    line_handle_list = list()
    legend_label_list = list()
    for sample_counter, i_sample in enumerate(selected_samples):
        cur_color = [grayscale_list[sample_counter], grayscale_list[sample_counter], grayscale_list[sample_counter]]
        line_handle, = cur_ax.plot(mo_obj_val_per_sample[i_sample, 0, :], mo_obj_val_per_sample[i_sample, 1, :], \
                                linestyle='-', marker='o', markersize=node_size, linewidth=line_width, color=cur_color)
        
        for i_mo_sol in range(0,n_mo_sol):
            _, = cur_ax.plot([mo_obj_val_per_sample[i_sample, 0, i_mo_sol]], [mo_obj_val_per_sample[i_sample, 1, i_mo_sol]], \
                                linestyle='', marker='o', markersize=node_size, linewidth=line_width, color=color_list(i_mo_sol), zorder=10)
        
        pi_factor = float(inputs[i_sample]/np.pi)
        if pi_factor == 0.25:
            legend_label = r'$\frac{1}{4}\pi$'
        elif pi_factor == 0.75:
            legend_label = r'$\frac{3}{4}\pi$'
        else:
            legend_label = r'$' + str(round(pi_factor,2)) + '\pi$'
        legend_label_list.append(legend_label)
        line_handle_list.append(line_handle)       

    cur_ax.set_xticks(ytick_choices[0])
    cur_ax.set_yticks(ytick_choices[1])
    plt.xlabel(label_list[0], fontsize=font_size)
    plt.ylabel(label_list[1], fontsize=font_size)
    cur_ax.tick_params(axis='both', which='both', labelsize=font_size)
    plt.legend(line_handle_list, legend_label_list, loc='upper right', fontsize=legend_font_size)
    plt.savefig(os.path.join(output_folder, '{}d_approximated_pareto_fronts.png'.format(n_mo_obj)), bbox_inches="tight")
    plt.close(fig)


def plot_os_pareto_fronts_3D(cfg, output_dict, output_folder):
    """
    plot the NNs' positions in objective space
    """
    color_list = plt.cm.get_cmap('tab10',10)
    node_size = 15
    line_width = 5
    font_size = 22
    legend_font_size = 22

    ytick_choices = []
    yticklabel_choices = []
    ytick_choices.append([0, 1, 2, 3])
    ytick_choices.append([0, 0.5, 1, 1.5])
    ytick_choices.append([0, 1, 2, 3])

    label_list = [r'$(\sin(x)-z)^{2}$',r'$(\cos(x)-z)^{2}$',r'$(\sin(x+\pi)-z)^{2}$'] # changed label order to match changed dimension order when plotting: 1 0 2

    inputs = output_dict["validation_data_x"]
    mo_obj_val_per_sample = output_dict["validation_loss"] #n_samples * n_mo_obj * n_mo_sol
    n_samples, n_mo_obj, n_mo_sol = mo_obj_val_per_sample.shape

    # sort inputs in increasing order
    sort_indices = np.argsort(inputs)
    inputs = inputs[sort_indices]
    mo_obj_val_per_sample = mo_obj_val_per_sample[sort_indices, :, :]

    selected_samples= np.arange(20, 100, 15) # 0.25pi to 0.55pi
    grayscale_list = np.linspace(0, 0.75, len(selected_samples))

    fig,cur_ax = plt.subplots(figsize=(10,10), dpi=100)
    cur_ax.axis('off')
    cur_ax = fig.add_subplot(111, projection='3d')
    
    line_handle_list = list()
    legend_label_list = list()
    for sample_counter,i_sample in enumerate(selected_samples):
        cur_color = [grayscale_list[sample_counter], grayscale_list[sample_counter], grayscale_list[sample_counter]]
        # ---- sorting so that the line is plotted in straight direction ----
        # ---- also plotting sine loss first, becasue the front looks more understandable this way ----
        sort_indices = np.argsort(mo_obj_val_per_sample[i_sample, 1, :])
        line_handle, = cur_ax.plot(mo_obj_val_per_sample[i_sample, 1, sort_indices], mo_obj_val_per_sample[i_sample, 0, sort_indices],\
                                     mo_obj_val_per_sample[i_sample, 2, sort_indices], linestyle='-', marker='o', markersize=node_size, \
                                    linewidth=line_width, color=cur_color) # changed dimension order when plotting: 1 0 2
        
        pi_factor = float(inputs[i_sample]/np.pi)
        if pi_factor == 0.25:
            legend_label = r'$\frac{1}{4}\pi$'
        elif pi_factor == 0.75:
            legend_label = r'$\frac{3}{4}\pi$'
        else:
            legend_label = r'$' + str(round(pi_factor,2)) + '\pi$'
        legend_label_list.append(legend_label)
        line_handle_list.append(line_handle)       

        for i_mo_sol in range(0,n_mo_sol):
            _, = cur_ax.plot([mo_obj_val_per_sample[i_sample, 1, i_mo_sol]], [mo_obj_val_per_sample[i_sample, 0, i_mo_sol]], \
                             [mo_obj_val_per_sample[i_sample, 2, i_mo_sol]], linestyle='', marker='o', markersize=node_size, \
                                linewidth=line_width, color=color_list(i_mo_sol)) # changed dimension order when plotting: 1 0 2
    
    cur_ax.set_xticks(ytick_choices[0])
    cur_ax.set_yticks(ytick_choices[1])
    cur_ax.set_zticks(ytick_choices[2])
    cur_ax.set_xlabel(label_list[0], labelpad=12, fontsize=font_size)
    cur_ax.set_ylabel(label_list[1], labelpad=14, fontsize=font_size)
    cur_ax.set_zlabel(label_list[2], labelpad=12, fontsize=font_size)
    cur_ax.tick_params(axis='both', which='both', labelsize=font_size)

    cur_ax.set_xlim([-0.2, 3.75])
    cur_ax.set_ylim([-0.2, 2.00])
    cur_ax.set_zlim([-0.2, 3.75])

    plt.legend(line_handle_list, legend_label_list, loc='upper left', bbox_to_anchor=(1,1), fontsize=legend_font_size)
    plt.savefig(os.path.join(output_folder, '{}d_approximated_pareto_fronts.png'.format(n_mo_obj)), bbox_inches="tight")
    plt.close(fig)


def plot_os_pareto_fronts(cfg, output_dict, output_folder):
    """
    plot the NNs' positions in objective space
    """
    mo_obj_val_per_sample = output_dict["validation_loss"] #n_samples * n_mo_obj * n_mo_sol
    n_samples, n_mo_obj, n_mo_sol = mo_obj_val_per_sample.shape

    if n_mo_obj==2:
        plot_os_pareto_fronts_2D(cfg, output_dict, output_folder)
    elif n_mo_obj==3:
        plot_os_pareto_fronts_3D(cfg, output_dict, output_folder)
    else:
        raise NotImplementedError("plotting implemented only for 2 or 3 objectives.")


