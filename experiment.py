import os
import time
import random
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

# --- Classical Machine Learning Baselines ---
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# --- Deep Learning Framework ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- Resampling Algorithms (Imbalanced Learning) ---
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")


# ================= 0. Fix Global Random Seeds =================
def seed_everything(seed=42):
    """Ensure reproducibility by fixing all random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ================= 1. Global Configurations =================
class Config:
    CSV_PATH = 'data.csv'
    LOG_DIR = 'ablation_logs'
    TUNING_RESULT_PATH = 'hyperparameter_tuning_results.csv'
    BEST_MODEL_PATH = 'best_dual_cnn_model.pth'

    MIN_SAMPLES = 10
    MAX_LEN_DYN = 150
    MAX_LEN_STA = 50

    EPOCHS = 25
    WEIGHT_DECAY = 0.0001
    NUM_CHANNELS = 128
    FOCAL_GAMMA = 2.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_FOLDS = 5


def setup_logger(log_dir):
    """Set up the logger to output to both console and file."""
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"experiment_full_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = logging.getLogger("Experiment")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(log_file))
        logger.addHandler(logging.StreamHandler())
    return logger


logger = setup_logger(Config.LOG_DIR)


# ================= 2. Data Processing & Word Embeddings =================
def process_raw_sequences(text_series):
    """Tokenize raw text sequences."""
    return [str(text).split() for text in text_series]


def train_word2vec_embeddings(all_sequences, embed_dim):
    """Generate basic random word embeddings (simulating Word2Vec for demonstration)."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in all_sequences:
        for word in seq:
            if word not in vocab: vocab[word] = len(vocab)
    weight_matrix = np.random.normal(size=(len(vocab), embed_dim))
    weight_matrix[0] = np.zeros(embed_dim)  # Padding index is 0
    return vocab, weight_matrix


def seq_to_embedded_matrix(sequences, vocab, weights, max_len):
    """Convert sequences of words to fixed-length embedded matrices."""
    n_samples = len(sequences)
    embed_dim = weights.shape[1]
    embedded_matrix = np.zeros((n_samples, max_len, embed_dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, w in enumerate(seq[:max_len]):
            idx = vocab.get(w, vocab["<UNK>"])
            embedded_matrix[i, j, :] = weights[idx]
    return embedded_matrix


# ================= Algorithm 4-1: Hybrid Resampling =================
def algorithm_4_1_resample(X_train_flat, y_train):
    """Balance the dataset using a hybrid approach of SMOTE and RandomUnderSampler."""
    counts = Counter(y_train)
    N = len(y_train)
    C = len(counts)
    if C == 0: return X_train_flat, y_train
    N_resample = int(N / C)

    # Identify classes to oversample (SMOTE) and undersample (RUS)
    smote_dict = {k: N_resample for k, v in counts.items() if v < N_resample}
    under_dict = {k: N_resample for k, v in counts.items() if v > N_resample}

    X_res, y_res = X_train_flat, y_train
    if smote_dict:
        min_samples = min([counts[k] for k in smote_dict.keys()])
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        try:
            smote = SMOTE(sampling_strategy=smote_dict, k_neighbors=max(1, k_neighbors), random_state=42)
            X_res, y_res = smote.fit_resample(X_res, y_res)
        except ValueError:
            pass

    if under_dict:
        rus = RandomUnderSampler(sampling_strategy=under_dict, random_state=42)
        X_res, y_res = rus.fit_resample(X_res, y_res)
    return X_res, y_res


# ================= 3. Core Models & Components =================
class ImprovedGatedMultimodalFusion(nn.Module):
    """Gated attention mechanism to fuse dynamic and static modalities dynamically."""

    def __init__(self, dyn_dim, sta_dim):
        super().__init__()
        self.bn_dyn = nn.BatchNorm1d(dyn_dim)
        self.bn_sta = nn.BatchNorm1d(sta_dim)
        combined_dim = dyn_dim + sta_dim
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2), nn.BatchNorm1d(combined_dim // 2),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(combined_dim // 2, 2), nn.Sigmoid()
        )

    def forward(self, f_dyn, f_sta):
        f_dyn_norm, f_sta_norm = self.bn_dyn(f_dyn), self.bn_sta(f_sta)
        combined = torch.cat([f_dyn_norm, f_sta_norm], dim=1)
        weights = self.attention(combined)
        return torch.cat([f_dyn_norm * weights[:, 0].unsqueeze(1), f_sta_norm * weights[:, 1].unsqueeze(1)], dim=1)


class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance during training."""

    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma, self.reduction = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


class ProposedDualCNN(nn.Module):
    """The Proposed Dual-Branch CNN for Multimodal Sequence Classification."""

    def __init__(self, num_classes, embed_dim, sta_kernels, dyn_kernels, dyn_dilations, dropout_rate=0.5, mode='dual',
                 fusion_type='gated'):
        super().__init__()
        self.mode = mode
        self.fusion_type = fusion_type

        # Dynamic Branch (Dilated CNNs to capture long-term API dependencies)
        self.convs_dyn = nn.ModuleList(
            [nn.Conv1d(embed_dim, Config.NUM_CHANNELS, k, padding=d * (k - 1) // 2, dilation=d) for k, d in
             zip(dyn_kernels, dyn_dilations)])
        # Static Branch (Standard CNNs for local structural features)
        self.convs_sta = nn.ModuleList(
            [nn.Conv1d(embed_dim, Config.NUM_CHANNELS, k, padding=k // 2) for k in sta_kernels])
        self.dropout = nn.Dropout(dropout_rate)

        dyn_out_dim = Config.NUM_CHANNELS * len(dyn_kernels)
        sta_out_dim = Config.NUM_CHANNELS * len(sta_kernels)

        # Configure fusion and FC layers based on ablation mode
        if self.mode == 'dual':
            if self.fusion_type == 'gated': self.fusion = ImprovedGatedMultimodalFusion(dyn_out_dim, sta_out_dim)
            fc1_in_dim = dyn_out_dim + sta_out_dim
        elif self.mode == 'dyn_only':
            fc1_in_dim = dyn_out_dim
        elif self.mode == 'sta_only':
            fc1_in_dim = sta_out_dim

        self.fc1 = nn.Linear(fc1_in_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward_branch(self, x, convs):
        x = x.permute(0, 2, 1)
        conved = [F.relu(conv(x)) for conv in convs]
        pooled = [F.adaptive_max_pool1d(c, 1).squeeze(2) for c in conved]
        return torch.cat(pooled, dim=1)

    def forward(self, dyn_emb, sta_emb):
        if self.mode == 'dual':
            f_dyn = self.forward_branch(dyn_emb, self.convs_dyn)
            f_sta = self.forward_branch(sta_emb, self.convs_sta)
            if self.fusion_type == 'gated':
                fused_features = self.fusion(f_dyn, f_sta)
            else:
                fused_features = torch.cat([f_dyn, f_sta], dim=1)
        elif self.mode == 'dyn_only':
            fused_features = self.forward_branch(dyn_emb, self.convs_dyn)
        elif self.mode == 'sta_only':
            fused_features = self.forward_branch(sta_emb, self.convs_sta)

        x = F.relu(self.bn1(self.fc1(self.dropout(fused_features))))
        x = F.relu(self.fc2(self.dropout(x)))
        return self.out(x)


# ================= 4. Deep Learning Baselines =================
class BaselineCNN(nn.Module):
    def __init__(self, num_classes, embed_dim, kernels):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, Config.NUM_CHANNELS, k, padding=k // 2) for k in kernels])
        self.fc1, self.out = nn.Linear(Config.NUM_CHANNELS * len(kernels), 128), nn.Linear(128, num_classes)

    def forward(self, dyn_emb, sta_emb):
        x = torch.cat([dyn_emb, sta_emb], dim=1).permute(0, 2, 1)
        x = torch.cat([F.adaptive_max_pool1d(F.relu(conv(x)), 1).squeeze(2) for conv in self.convs], dim=1)
        return self.out(F.relu(self.fc1(x)))


class BaselineLSTM(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, dyn_emb, sta_emb):
        _, (hn, _) = self.lstm(torch.cat([dyn_emb, sta_emb], dim=1))
        return self.fc(torch.cat([hn[-2], hn[-1]], dim=1))


class BaselineMLP(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear((Config.MAX_LEN_DYN + Config.MAX_LEN_STA) * embed_dim, 512), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, num_classes))

    def forward(self, dyn_emb, sta_emb):
        x = torch.cat([dyn_emb, sta_emb], dim=1)
        return self.net(x.view(x.size(0), -1))


# ================= 5. Core Experiment Engine & Utility Functions =================
def evaluate_efficiency_and_params(model, model_type, embed_dim, device):
    """Evaluate model parameters, peak memory usage, and inference latency."""
    dummy_dyn = torch.FloatTensor(np.random.randn(1, Config.MAX_LEN_DYN, embed_dim).astype(np.float32)).to(device)
    dummy_sta = torch.FloatTensor(np.random.randn(1, Config.MAX_LEN_STA, embed_dim).astype(np.float32)).to(device)
    dummy_flat_np = np.hstack([dummy_dyn.cpu().numpy().reshape(1, -1), dummy_sta.cpu().numpy().reshape(1, -1)])
    peak_mem, total_params, lat = "N/A", "N/A", 0

    if model_type == 'dl':
        model.eval()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy_dyn, dummy_sta)
            peak_mem = f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
        total_params = f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M"

        # Latency Warm-up & Measurement
        with torch.no_grad():
            for _ in range(10): _ = model(dummy_dyn, dummy_sta)
            start = time.time()
            for _ in range(100): _ = model(dummy_dyn, dummy_sta)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat = (time.time() - start) / 100 * 1000
    else:
        # Machine Learning Latency Measurement
        for _ in range(2): _ = model.predict(dummy_flat_np)
        start = time.time()
        for _ in range(50): _ = model.predict(dummy_flat_np)
        lat = (time.time() - start) / 50 * 1000

    return total_params, peak_mem, f"{lat:.2f} ms"


def run_ablation_experiment(exp_config, X_dyn, X_sta, y, num_classes, is_tuning=False):
    """Execute a single ablation/tuning experiment using K-Fold Cross Validation."""
    if not is_tuning: logger.info(f"▶️ Running Experiment: [{exp_config['name']}]")

    embed_dim = exp_config['embed_dim']
    N_SAMPLES, DYN_FLAT_DIM = X_dyn.shape[0], Config.MAX_LEN_DYN * embed_dim
    X_concat_flat = np.hstack([X_dyn.reshape(N_SAMPLES, -1), X_sta.reshape(N_SAMPLES, -1)])
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_concat_flat, y)):
        X_tr_flat, y_tr = X_concat_flat[t_idx], y[t_idx]

        use_resample = exp_config.get('use_resample', True)
        if use_resample:
            X_tr_flat_res, y_tr_res = algorithm_4_1_resample(X_tr_flat, y_tr)
        else:
            X_tr_flat_res, y_tr_res = X_tr_flat, y_tr

        X_tr_d = X_tr_flat_res[:, :DYN_FLAT_DIM].reshape(-1, Config.MAX_LEN_DYN, embed_dim)
        X_tr_s = X_tr_flat_res[:, DYN_FLAT_DIM:].reshape(-1, Config.MAX_LEN_STA, embed_dim)

        loader_tr = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_d), torch.FloatTensor(X_tr_s), torch.LongTensor(y_tr_res)),
            batch_size=exp_config['batch_size'], shuffle=True)
        loader_val = DataLoader(
            TensorDataset(torch.FloatTensor(X_dyn[v_idx]), torch.FloatTensor(X_sta[v_idx]), torch.LongTensor(y[v_idx])),
            batch_size=exp_config['batch_size'])

        use_dilation = exp_config.get('use_dilation', True)
        current_dilations = exp_config['dyn_dilations'] if use_dilation else [1] * len(exp_config['dyn_kernels'])

        model = ProposedDualCNN(
            num_classes, embed_dim, exp_config['sta_kernels'], exp_config['dyn_kernels'],
            current_dilations, exp_config['dropout'], mode=exp_config['mode'],
            fusion_type=exp_config.get('fusion_type', 'gated')
        ).to(Config.DEVICE)

        opt = optim.Adam(model.parameters(), lr=exp_config['lr'], weight_decay=Config.WEIGHT_DECAY)

        use_focal = exp_config.get('use_focal', True)
        crit = FocalLoss(gamma=Config.FOCAL_GAMMA).to(Config.DEVICE) if use_focal else nn.CrossEntropyLoss().to(
            Config.DEVICE)

        best_fold_preds, best_fold_acc = [], 0
        for epoch in range(Config.EPOCHS):
            model.train()
            for db, sb, yb in loader_tr:
                db, sb, yb = db.to(Config.DEVICE), sb.to(Config.DEVICE), yb.to(Config.DEVICE)
                opt.zero_grad()
                loss = crit(model(db, sb), yb)
                loss.backward()
                opt.step()

            model.eval()
            curr_p = []
            with torch.no_grad():
                for db, sb, yb in loader_val:
                    curr_p.extend(torch.max(model(db.to(Config.DEVICE), sb.to(Config.DEVICE)), 1)[1].cpu().numpy())
            acc = accuracy_score(y[v_idx], curr_p)
            if acc >= best_fold_acc:
                best_fold_acc, best_fold_preds = acc, curr_p

        all_y_true.extend(y[v_idx])
        all_y_pred.extend(best_fold_preds)

    return accuracy_score(all_y_true, all_y_pred), precision_score(all_y_true, all_y_pred, average='weighted',
                                                                   zero_division=0), recall_score(all_y_true,
                                                                                                  all_y_pred,
                                                                                                  average='weighted',
                                                                                                  zero_division=0), f1_score(
        all_y_true, all_y_pred, average='weighted', zero_division=0), all_y_true, all_y_pred


def train_eval_baseline(model_name, model_class, model_type, X_dyn, X_sta, y, num_classes, embed_dim, opt_batch,
                        opt_lr):
    """Train and evaluate baseline ML and DL models."""
    logger.info(f"▶️ Training and evaluating baseline: {model_name} ...")
    N_SAMPLES, DYN_FLAT = X_dyn.shape[0], Config.MAX_LEN_DYN * embed_dim
    X_concat_flat = np.hstack([X_dyn.reshape(N_SAMPLES, -1), X_sta.reshape(N_SAMPLES, -1)])
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []
    trained_model = None

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_concat_flat, y)):
        X_tr_flat, y_tr = X_concat_flat[t_idx], y[t_idx]
        X_tr_flat_res, y_tr_res = algorithm_4_1_resample(X_tr_flat, y_tr)

        if model_type == 'ml':
            model = model_class()
            model.fit(X_tr_flat_res, y_tr_res)
            preds = model.predict(X_concat_flat[v_idx])
            trained_model = model
        elif model_type == 'dl':
            X_tr_d = X_tr_flat_res[:, :DYN_FLAT].reshape(-1, Config.MAX_LEN_DYN, embed_dim)
            X_tr_s = X_tr_flat_res[:, DYN_FLAT:].reshape(-1, Config.MAX_LEN_STA, embed_dim)
            loader_tr = DataLoader(
                TensorDataset(torch.FloatTensor(X_tr_d), torch.FloatTensor(X_tr_s), torch.LongTensor(y_tr_res)),
                batch_size=opt_batch, shuffle=True)
            loader_val = DataLoader(TensorDataset(torch.FloatTensor(X_dyn[v_idx]), torch.FloatTensor(X_sta[v_idx])),
                                    batch_size=opt_batch)
            model = model_class(num_classes, embed_dim).to(Config.DEVICE)
            opt = optim.Adam(model.parameters(), lr=opt_lr)
            crit = nn.CrossEntropyLoss()

            for _ in range(15):
                model.train()
                for db, sb, yb in loader_tr:
                    db, sb, yb = db.to(Config.DEVICE), sb.to(Config.DEVICE), yb.to(Config.DEVICE)
                    opt.zero_grad()
                    loss = crit(model(db, sb), yb)
                    loss.backward()
                    opt.step()
            model.eval()
            preds = []
            with torch.no_grad():
                for db, sb in loader_val: preds.extend(
                    torch.max(model(db.to(Config.DEVICE), sb.to(Config.DEVICE)), 1)[1].cpu().numpy())
            trained_model = model

        all_y_true.extend(y[v_idx])
        all_y_pred.extend(preds)

    return accuracy_score(all_y_true, all_y_pred), precision_score(all_y_true, all_y_pred, average='weighted',
                                                                   zero_division=0), recall_score(all_y_true,
                                                                                                  all_y_pred,
                                                                                                  average='weighted',
                                                                                                  zero_division=0), f1_score(
        all_y_true, all_y_pred, average='weighted', zero_division=0), *evaluate_efficiency_and_params(trained_model,
                                                                                                      model_type,
                                                                                                      embed_dim,
                                                                                                      Config.DEVICE)


def train_and_save_final_model(X_dyn, X_sta, y, num_classes, embed_dim, opt_dyn_k, opt_sta_k, opt_dyn_d, opt_drop,
                               opt_batch, opt_lr, save_path):
    """Train the final model using the entire balanced dataset and save weights."""
    logger.info(f"\n💾 [Model Saving]: Training final model on FULL balanced dataset and saving to {save_path}")
    N_SAMPLES, DYN_FLAT_DIM = X_dyn.shape[0], Config.MAX_LEN_DYN * embed_dim
    X_res_flat, y_res = algorithm_4_1_resample(np.hstack([X_dyn.reshape(N_SAMPLES, -1), X_sta.reshape(N_SAMPLES, -1)]),
                                               y)
    X_res_dyn = X_res_flat[:, :DYN_FLAT_DIM].reshape(-1, Config.MAX_LEN_DYN, embed_dim)
    X_res_sta = X_res_flat[:, DYN_FLAT_DIM:].reshape(-1, Config.MAX_LEN_STA, embed_dim)

    model = ProposedDualCNN(num_classes, embed_dim, opt_sta_k, opt_dyn_k, opt_dyn_d, opt_drop, mode='dual',
                            fusion_type='gated').to(Config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=opt_lr, weight_decay=Config.WEIGHT_DECAY)
    crit = FocalLoss(gamma=Config.FOCAL_GAMMA).to(Config.DEVICE)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_res_dyn), torch.FloatTensor(X_res_sta), torch.LongTensor(y_res)),
        batch_size=opt_batch, shuffle=True)

    model.train()
    for _ in range(Config.EPOCHS):
        for db, sb, yb in loader:
            opt.zero_grad()
            crit(model(db.to(Config.DEVICE), sb.to(Config.DEVICE)), yb.to(Config.DEVICE)).backward()
            opt.step()

    torch.save(model.state_dict(), save_path)
    logger.info("✅ Best model weights saved successfully!")


# ================= 6. Main Execution Pipeline =================
def main():
    logger.info(f"⚙️ Automated Pipeline Initiated | Environment: {Config.DEVICE}")
    try:
        df = pd.read_csv(Config.CSV_PATH).fillna('')
    except FileNotFoundError:
        logger.warning("Dataset not found. Generating dummy dataset for testing purposes.")
        df = pd.DataFrame({'apt_label': np.random.choice(['apt1', 'apt2', 'apt3'], 200, p=[0.7, 0.2, 0.1]),
                           'api_sequence': [' '.join(['api' + str(np.random.randint(10)) for _ in range(50)]) for _ in
                                            range(200)],
                           'static_sequence': [' '.join(['func' + str(np.random.randint(10)) for _ in range(20)]) for _
                                               in range(200)]})

    valid_labels = df['apt_label'].value_counts()[df['apt_label'].value_counts() >= Config.MIN_SAMPLES].index
    df = df[df['apt_label'].isin(valid_labels)].reset_index(drop=True)

    # Generate and print Mapping Dict
    le = LabelEncoder()
    y = le.fit_transform(df['apt_label'])
    target_names = le.classes_
    mapping_dict = {i: name for i, name in enumerate(target_names)}

    print("\n" + "=" * 50)
    print("🏷️ Label Mapping Dictionary:")
    print(mapping_dict)
    print("=" * 50 + "\n")
    logger.info(f"Label Mapping Dictionary: {mapping_dict}")

    num_classes = len(target_names)
    dyn_seqs, sta_seqs = process_raw_sequences(df['api_sequence']), process_raw_sequences(df['static_sequence'])

    # ---------------- Phase 1: Advanced Dynamic Grid Search ----------------
    logger.info("\n" + "=" * 125)
    logger.info("🔍 [Phase 1]: Hyperparameter Grid Search (Embeddings & Dropout included)")

    embed_dims_to_test = [50, 100]
    dropouts_to_test = [0.3, 0.5]
    batch_sizes_to_test = [32, 64]
    learning_rates_to_test = [1e-3, 5e-4]
    kernel_pairs_to_test = [{'dyn': [3, 3, 3], 'sta': [3, 3, 3]}, {'dyn': [3, 4, 5], 'sta': [2, 3, 4]}]
    opt_dyn_d = [1, 2, 4]

    data_cache = {}
    for edim in embed_dims_to_test:
        vocab, weights = train_word2vec_embeddings(dyn_seqs + sta_seqs, edim)
        data_cache[edim] = (seq_to_embedded_matrix(dyn_seqs, vocab, weights, Config.MAX_LEN_DYN),
                            seq_to_embedded_matrix(sta_seqs, vocab, weights, Config.MAX_LEN_STA))

    tuning_results = []
    best_f1 = 0.0
    best_y_true, best_y_pred = [], []
    opt_embed, opt_drop, opt_batch, opt_lr, opt_dyn_k, opt_sta_k = 100, 0.3, 64, 1e-3, [3, 3, 3], [3, 3, 3]
    opt_X_dyn, opt_X_sta = data_cache[100]

    logger.info(f"{'Dim':<4} | {'Drop':<4} | {'BS':<4} | {'LR':<6} | {'Dyn K.':<12} | {'Sta K.':<12} | {'F1':<6}")
    logger.info("-" * 80)

    for edim in embed_dims_to_test:
        X_dyn_curr, X_sta_curr = data_cache[edim]
        for drop in dropouts_to_test:
            for bs in batch_sizes_to_test:
                for lr in learning_rates_to_test:
                    for pair in kernel_pairs_to_test:
                        config = {
                            "name": f"Tuning", "mode": "dual", "fusion_type": "gated",
                            "embed_dim": edim, "sta_kernels": pair['sta'], "dyn_kernels": pair['dyn'],
                            "dyn_dilations": opt_dyn_d, "dropout": drop, "batch_size": bs, "lr": lr,
                            "use_resample": True, "use_focal": True, "use_dilation": True
                        }

                        _, _, _, f1, y_true_curr, y_pred_curr = run_ablation_experiment(config, X_dyn_curr, X_sta_curr,
                                                                                        y, num_classes, is_tuning=True)
                        tuning_results.append(
                            {'embed_dim': edim, 'dropout': drop, 'batch_size': bs, 'lr': lr, 'dyn_k': str(pair['dyn']),
                             'f1': f1})
                        logger.info(
                            f"{edim:<4} | {drop:<4} | {bs:<4} | {lr:<6} | {str(pair['dyn']):<12} | {str(pair['sta']):<12} | {f1:.4f}")

                        if f1 > best_f1:
                            best_f1 = f1
                            best_y_true = y_true_curr
                            best_y_pred = y_pred_curr
                            opt_embed, opt_drop, opt_batch, opt_lr = edim, drop, bs, lr
                            opt_dyn_k, opt_sta_k = pair['dyn'], pair['sta']
                            opt_X_dyn, opt_X_sta = X_dyn_curr, X_sta_curr

    pd.DataFrame(tuning_results).to_csv(Config.TUNING_RESULT_PATH, index=False)
    logger.info(
        f"✅ Grid Search Complete! Optimal -> Dim:{opt_embed}, Drop:{opt_drop}, BS:{opt_batch}, LR:{opt_lr}, DynK:{opt_dyn_k}, StaK:{opt_sta_k} (F1:{best_f1:.4f})")

    # ---------------- Phase 1.2: Optimal Model Metrics & Confusion Matrix ----------------
    logger.info("\n" + "=" * 125)
    logger.info("📊 [Phase 1.2]: Detailed Class-wise Metrics and Confusion Matrix (Optimal Parameters)")

    # Print class-wise Precision, Recall, F1
    report = classification_report(best_y_true, best_y_pred, target_names=target_names, digits=4)
    logger.info("\nDetailed Classification Report (Precision, Recall, F1-score):\n" + report)

    # Calculate separate accuracy for each APT group
    cm = confusion_matrix(best_y_true, best_y_pred)
    logger.info("Individual Accuracy per APT group:")
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        class_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        logger.info(f"  - Group {target_names[i]}: {class_acc:.4f}")

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix of the Best Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = 'best_model_confusion_matrix.png'
    plt.savefig(cm_path)
    logger.info(f"\n✅ Confusion matrix plotted and saved as '{cm_path}'")

    # Train and save final model
    train_and_save_final_model(opt_X_dyn, opt_X_sta, y, num_classes, opt_embed, opt_dyn_k, opt_sta_k, opt_dyn_d,
                               opt_drop, opt_batch, opt_lr, Config.BEST_MODEL_PATH)

    # ---------------- Phase 2: Component Ablation Study ----------------
    logger.info("\n" + "=" * 125)
    logger.info("🔬 [Phase 2]: Component Ablation Study")

    base_cfg = {"embed_dim": opt_embed, "sta_kernels": opt_sta_k, "dyn_kernels": opt_dyn_k, "dyn_dilations": opt_dyn_d,
                "dropout": opt_drop, "batch_size": opt_batch, "lr": opt_lr}

    ablation_configs = [
        {**base_cfg, "name": "0. Full Model (Ours)", "mode": "dual", "fusion_type": "gated", "use_resample": True,
         "use_focal": True, "use_dilation": True},
        {**base_cfg, "name": "1a. w/o Static (Dynamic Only)", "mode": "dyn_only", "fusion_type": "gated",
         "use_resample": True, "use_focal": True, "use_dilation": True},
        {**base_cfg, "name": "1b. w/o Dynamic (Static Only)", "mode": "sta_only", "fusion_type": "gated",
         "use_resample": True, "use_focal": True, "use_dilation": True},
        {**base_cfg, "name": "2. w/o Gated Fusion (Simple Concat)", "mode": "dual", "fusion_type": "concat",
         "use_resample": True, "use_focal": True, "use_dilation": True}
    ]

    res_ablation = []
    for exp in ablation_configs:
        acc, prec, rec, f1, _, _ = run_ablation_experiment(exp, opt_X_dyn, opt_X_sta, y, num_classes)
        res_ablation.append([exp['name'], acc, prec, rec, f1])

    logger.info("\n" + "-" * 90)
    logger.info(f"{'Ablation Variant'.ljust(40)} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1-Score':<6}")
    logger.info("-" * 90)
    for name, acc, prec, rec, f1 in res_ablation:
        logger.info(f"{name.ljust(40)} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}")
    logger.info("-" * 90)

    # ---------------- Phase 3: Efficiency Evaluation ----------------
    logger.info("\n" + "=" * 125)
    logger.info("⚡ [Phase 3]: Model Efficiency and Resource Consumption Evaluation")
    best_model = ProposedDualCNN(num_classes, opt_embed, opt_sta_k, opt_dyn_k, opt_dyn_d, opt_drop, mode='dual',
                                 fusion_type='gated').to(Config.DEVICE)
    params_ours, mem_ours, lat_ours = evaluate_efficiency_and_params(best_model, 'dl', opt_embed, Config.DEVICE)
    logger.info(f"Ours -> Parameters: {params_ours} | Peak Memory: {mem_ours} | Latency: {lat_ours}")
    logger.info("=" * 125)

    # ---------------- Phase 4: Baseline Comparisons ----------------
    logger.info("\n" + "=" * 125)
    logger.info("⚔️ [Phase 4]: Baseline Comparisons (Using Optimal Features)")
    logger.info("=" * 125)

    baselines = {
        "CNN (Best Dyn Kernels)": (lambda c, d: BaselineCNN(c, d, opt_dyn_k), 'dl'),
        "CNN (Best Sta Kernels)": (lambda c, d: BaselineCNN(c, d, opt_sta_k), 'dl'),
        "LSTM (Baseline)": (BaselineLSTM, 'dl'), "MLP (Baseline)": (BaselineMLP, 'dl'),
        "SVM (SVC)": (SVC, 'ml'), "Decision Tree": (DecisionTreeClassifier, 'ml'),
        "Naive Bayes": (GaussianNB, 'ml'), "KNN (k=5)": (KNeighborsClassifier, 'ml'),
        "Proposed Dual-CNN (Ours)": (
            lambda c, d: ProposedDualCNN(c, d, opt_sta_k, opt_dyn_k, opt_dyn_d, opt_drop, mode='dual',
                                         fusion_type='gated'), 'dl')
    }

    results = []
    for name, (model_class, m_type) in baselines.items():
        try:
            acc, prec, rec, f1, params, mem, lat = train_eval_baseline(name, model_class, m_type, opt_X_dyn, opt_X_sta,
                                                                       y, num_classes, opt_embed, opt_batch, opt_lr)
            results.append([name, acc, prec, rec, f1, params, mem, lat])
        except Exception as e:
            logger.warning(f"❌ Model {name} failed: {str(e)}")

    logger.info("\n" + "=" * 125)
    logger.info(
        f"{'Model Name'.ljust(35)} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Params':<10} | {'Mem':<10} | {'Lat'}")
    logger.info("-" * 125)
    for name, acc, prec, rec, f1, params, mem, lat in results:
        logger.info(
            f"{('✨ ' + name if 'Ours' in name else name).ljust(35)} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {params:<10} | {mem:<10} | {lat}")


if __name__ == "__main__":
    main()
