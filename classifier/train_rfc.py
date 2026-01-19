import numpy as np
from flaml import AutoML

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class RefKFold(KFold):
    def __init__(self, n_splits, classes_train, refs):
        super().__init__(n_splits)
        self.n_splits = n_splits
        refs

        test_ref_list = {i: [] for i in range(n_splits)}

        k = 0
        for i in range(3):
            cls_refs = classes_train[classes_train[:, 1] == i, 0]
            for ref in cls_refs:
                k = k % self.n_splits
                test_ref_list[k].append(ref)
                k += 1

        self.test_masks = np.zeros([self.n_splits, len(refs)]) == 1

        for j in range(self.n_splits):
            for m in range(len(refs)):
                if refs[m] in test_ref_list[j]:
                    self.test_masks[j][m] = True


    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        for i in range(self.n_splits):
            yield np.where(np.logical_not(self.test_masks[i])), np.where(self.test_masks[i])

def train_and_eval_single(features, y, train_mask, k_fold, feature_mask, classes_test, refs_test, feature_names):
    X = np.column_stack([features[i] for i in feature_mask])
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[np.logical_not(train_mask)]
    y_test = y[np.logical_not(train_mask)]

    automl = AutoML()
    automl.fit(X_train, y_train, task="classification", time_budget=240, skip_transform=False, metric='auto',
               split_type=k_fold, ensemble=True, log_type='better', seed=545468, verbose=False)

    y_pred = automl.predict(X_test)
    y_pred_proba = automl.predict_proba(X_test)

    print(5 * '*' + '+'.join([feature_names[i] for i in feature_mask]) + 5 * '*')

    print("Per frame accuracy")
    print(classification_report(y_test, y_pred, digits=3))

    frame_rep = classification_report(y_test, y_pred, digits=3, output_dict=True)

    voting = {k: np.zeros(3) for k in classes_test[:, 0]}

    for i in range(len(y_pred)):
        ref = refs_test[i]
        voting[ref] += y_pred_proba[i]

    pred_aggregated = []
    for ref in classes_test[:, 0]:
        pred_aggregated.append(np.argmax(voting[ref]))

    print("Per instance report")
    print(classification_report(classes_test[:, 1], pred_aggregated, digits=3))

    agg_rep = classification_report(classes_test[:, 1], pred_aggregated, digits=3, output_dict=True)

    result = {'model': automl,
              'frame_acc': frame_rep['accuracy'],
              'frame_macro_f1': frame_rep['macro avg']['f1-score'],
              'frame_micro_f1': frame_rep['weighted avg']['f1-score'],
              'agg_acc': agg_rep['accuracy'],
              'agg_macro_f1': agg_rep['macro avg']['f1-score'],
              'agg_micro_f1': agg_rep['weighted avg']['f1-score'],
        }

    return result


def print_results(results, feature_masks):
    print("Printing tex table")

    for feature_mask in feature_masks:
        result = results[feature_mask]
        line = [' ', ' ', ' ', ' ']
        for i in feature_mask:
            line[i] = '\\checkmark'

        if 0 in feature_mask:
            line.extend(['-', '-', '-'])
        else:
            line.extend([f'{100 * result[x]:.1f}' for x in ['frame_acc', 'frame_macro_f1', 'frame_micro_f1']])

        line.extend([f'{100 * result[x]:.1f}' for x in ['agg_acc', 'agg_macro_f1', 'agg_micro_f1']])

        print('&'.join(line) + '\\\\')


def eval_on_gsv(model, classes):
    features_gsv = np.genfromtxt('data/gsv_features_10.csv', skip_header=1, delimiter=',')

    refs = features_gsv[:, 0].astype(int)

    print(len(list(set(refs))))

    y_gsv = np.array([classes[x] for x in refs])
    X_box_gsv = features_gsv[:, 10:14]
    X_image_pca_gsv = features_gsv[:, 14:10 + 384]

    X_gsv = np.column_stack([X_box_gsv, X_image_pca_gsv[:, :3]])
    y_pred = model.predict(X_gsv)

    print("Per frame accuracy")
    print(classification_report(y_gsv, y_pred, digits=3))


def train_and_eval():
    classes_train = np.genfromtxt('data/train_IDs.csv', comments='#', delimiter=',', dtype=int)
    classes_test = np.genfromtxt('data/test_IDs.csv', comments='#', delimiter=',', dtype=int)

    use_for_train = {k: True for k in classes_train[:, 0]}
    use_for_train.update({k: False for k in classes_test[:, 0]})

    print(use_for_train[classes_test[0, 0]])

    classes = {x[0]: x[1] for x in classes_train}
    classes.update({x[0]: x[1] for x in classes_test})

    features = np.genfromtxt('data/all_features_10.csv', skip_header=1, delimiter=',')

    refs = features[:, 0].astype(int)

    y = np.array([classes[x] for x in refs])
    X_old = features[:, 3:10]
    X_box = features[:, 10:14]
    X_image_pca = features[:, 14:10 + 384]
    X_crop_pca = features[:, 14 + 384:10 + 2 * 384]
    # X_image = features[:, 14 + 2 * 384:10 + 3 * 384]
    # X_crop = features[:, 14 + 3 * 384:10 + 4 * 384]

    train_mask = [use_for_train[x] for x in refs]
    refs_test = refs[np.logical_not(train_mask)]
    k_fold = RefKFold(5, classes_train, refs[train_mask])

    features = [X_old, X_box, X_image_pca[:, :3], X_crop_pca[:, :3]]

    feature_masks = [
        (0,),
        (0, 1, 2, 3),
        (1,),
        (2,),
        (3,),
        (1,2),
        (1,3),
        (1,2,3),
        (2,3)
    ]

    feature_names = ['Orig.', 'B', 'I_full', 'I_crop']

    results = {}
    for feature_mask in feature_masks[-1:]:
        results[feature_mask] = train_and_eval_single(features, y, train_mask, k_fold, feature_mask, classes_test,
                                                      refs_test, feature_names)

    print_results(results, feature_masks[-1:])

    # eval_on_gsv(results[feature_masks[5]]['model'], classes)


if __name__ == '__main__':
    train_and_eval()