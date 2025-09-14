import os
import argparse
import warnings
import json
import shutil
from dataset import DataLoaderX as DataLoader
from options import DotDict
from exp_setup import log_path

def find_best_parametric(model_loader_fn, model_name, only_keep_best=False, ds_name='SCARED',
                          dataset=None, eval_kwargs=None, peft=True, pose_seq=1):
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dares')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    import evaluate_pose_and_intrinsics
    warnings.filterwarnings("ignore")
    if dataset is None:
        from exp_setup import ds_test
        dataset = ds_test
    model_path = os.path.join(log_path, model_name, 'models')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    opt_run = json.load(open(os.path.join(model_path, 'opt.json'), 'r'))
    opt_dict = DotDict(opt_run)
    def evaluate_model(weight_path, dataset=dataset, load_depth_from_npz=ds_name=='SCARED'):
        if ds_name == 'SCARED':
            import evaluate_depth_scared
            evaluate = evaluate_depth_scared.evaluate
        else:
            import evaluate_depth
            evaluate = evaluate_depth.evaluate
        test_dataloader = DataLoader(dataset, 16, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
        depth_model = model_loader_fn(opt_dict, weight_path, peft=peft)
        ds_and_model = {
            'dataloader': test_dataloader,
            'depth_model': depth_model,
            'output_dir': os.path.join(model_path, 'eval_images', model_name),
        }
        eval_args = dict(ds_and_model=ds_and_model, load_depth_from_npz=load_depth_from_npz)
        if eval_kwargs:
            eval_args.update(eval_kwargs)
        return evaluate(opt_dict, **eval_args)
    print(f"Testing {model_name}")
    weights = [w for w in os.listdir(model_path) if w.startswith('weights_')]
    if not weights:
        print(f"No weights found in {model_path} for model {model_name}.")
        return
    print(f"Found {len(weights)} weights for {model_name}")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    best_score = float('inf')
    best_weight = None
    best_pose_score = float('inf')
    best_pose_weight = None
    best_pose_ate = None
    METRIC = 'abs_rel'
    metric_indices = {
        'abs_rel': 0,
        'sq_rel': 1,
        'rmse': 2,
        'rmse_log': 3,
        'a1': 4,
        'a2': 5,
        'a3': 6
    }
    index = metric_indices[METRIC]
    pose_ate_dict = {}
    for weight in weights:
        weight_path = os.path.join(model_path, weight)
        # 深度评估 / Depth evaluation
        score = evaluate_model(weight_path)
        print(f"\t{weight}")
        if score[index] < best_score:
            best_score = score[index]
            best_weight = weight
        # 位姿/内参评估 / Pose/intrinsics evaluation
        try:
            pose_result = evaluate_pose_and_intrinsics.evaluate(
                opt=opt_dict,
                load_weights_folder=weight_path,
                dataset_name=ds_name,
                pose_seq=pose_seq,
            )
            # Use the returned score directly (lower is better)
            if pose_result and isinstance(pose_result, dict):
                # Extract ATE RMSE from the results dictionary
                if 'ate_rmse' in pose_result:
                    ate = pose_result['ate_rmse']
                    pose_ate_dict[weight] = ate
                else:
                    print(f"[WARN] No ATE RMSE found in pose evaluation results for {weight}")
            elif pose_result and isinstance(pose_result, (int, float)):
                ate = pose_result
                pose_ate_dict[weight] = ate
                if ate < best_pose_score:
                    best_pose_score = ate
                    best_pose_weight = weight
                    best_pose_ate = ate
        except Exception as e:
            print(f"Pose evaluation failed for {weight}: {e}")
    print(f"Best weight for {model_name} (depth): {best_weight} with score: {best_score}")
    print(f"Best weight for {model_name} (pose): {best_pose_weight} with ATE: {best_pose_ate}")
    print('-' * 50)
    # 只保留最佳深度和最佳位姿权重 / Only keep the best depth and pose weights
    best_weight_path = os.path.join(model_path, best_weight)
    best_pose_weight_path = os.path.join(model_path, best_pose_weight) if best_pose_weight else None
    best_dir = os.path.join(model_path, f'best_{ds_name}_depth')
    best_pose_dir = os.path.join(model_path, f'best_{ds_name}_pose')
    if only_keep_best:
        for weight in weights:
            weight_path = os.path.join(model_path, weight)
            if weight != best_weight and (best_pose_weight is None or weight != best_pose_weight):
                shutil.rmtree(weight_path)
        os.rename(best_weight_path, best_dir)
        if best_pose_weight_path and best_pose_weight != best_weight:
            os.rename(best_pose_weight_path, best_pose_dir)
    else:
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(best_weight_path, best_dir)
        if best_pose_weight_path and best_pose_weight != best_weight:
            if os.path.exists(best_pose_dir):
                shutil.rmtree(best_pose_dir)
            shutil.copytree(best_pose_weight_path, best_pose_dir)
    # 输出结果 / Output results
    result_file = os.path.join(model_path, 'best_results.txt')
    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            f.write('Best weight results:\n')
    with open(result_file, 'a') as f:
        f.write(f'Best weight path (depth): {best_weight} for dataset {ds_name}\n')
        f.write(f'Best weight path (pose): {best_pose_weight} for dataset {ds_name}\n')
        f.write('| Model Name    | abs_rel | sq_rel | rmse   | rmse_log | a1    | a2    | a3    |  ATE   |\n')
        f.write('|---------------|---------|--------|--------|----------|-------|-------|-------|--------|\n')
        f.write(f'| {model_name:<13} | {best_score:<7.4f} | {score[1]:<7.4f} | {score[2]:<7.4f} | {score[3]:<7.4f} | {score[4]:<7.4f} | {score[5]:<7.4f} | {score[6]:<7.4f} | {best_pose_ate if best_pose_ate is not None else "-":<6} |\n')
        # 输出pose最优模型的内参评估结果 / Output intrinsics evaluation results for the best pose model
        if best_pose_weight in pose_ate_dict:
            pose_result = None
            try:
                pose_result = evaluate_pose_and_intrinsics.evaluate(
                    opt=opt_dict,
                    load_weights_folder=os.path.join(model_path, best_pose_weight),
                    dataset_name=ds_name,
                    pose_seq=pose_seq,
                )
                if pose_result and isinstance(pose_result, dict):
                    if 'ate_rmse' in pose_result:
                        f.write(f"Best Pose Score (ATE RMSE): {pose_result['ate_rmse']:.6f}\n")
                    else:
                        f.write("[WARN] No ATE RMSE found in pose evaluation results\n")
                elif pose_result and isinstance(pose_result, (int, float)):
                    f.write(f"Best Pose Score: {pose_result:.6f}\n")
            except Exception as e:
                f.write(f"[WARN] Failed to evaluate best pose: {e}\n")
    
    print(f"Results appended to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific model with specified weights (parametric version).")
    parser.add_argument('--model_name', required=True, help="Name of the model directory under log_path.")
    args = parser.parse_args()
    from load_other_models import load_DARES
    model_loader_fn = lambda opt, w: load_DARES(opt, w, 'depth_model.pth', refine=False)
    find_best_parametric(model_loader_fn, args.model_name)
