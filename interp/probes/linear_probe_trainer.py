#!/usr/bin/env python3

import sys
import os
import json
import pickle
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from interp.representation_extractor import RepresentationExtractor

class ProbeDataset(Dataset):
    def __init__(self, representations: np.ndarray, labels: np.ndarray):
        self.representations = torch.FloatTensor(representations)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.representations[idx], self.labels[idx]

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class LinearProbeTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", model_type: str = "qwen3"):
        self.model_name = model_name
        self.model_type = model_type
        self.extractor = RepresentationExtractor(model_name, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = self._get_model_config(model_name)
    
    def _get_model_config(self, model_name: str) -> Dict:
        model_configs = {
            "Qwen/Qwen3-4B": {"num_layers": 32, "hidden_dim": 3584},
            "Qwen/Qwen3-8B": {"num_layers": 36, "hidden_dim": 4096}, 
            "Qwen/Qwen3-14B": {"num_layers": 40, "hidden_dim": 5120},
            "Qwen/Qwen3-32B": {"num_layers": 64, "hidden_dim": 5120}
        }
        
        if model_name in model_configs:
            return model_configs[model_name]
        else:
            print(f"Warning: Unknown model {model_name}, using default config")
            return {"num_layers": 36, "hidden_dim": 4096}
        
    def load_model(self):
        self.extractor.load_model()
        print(f"Model loaded. Found {len(self.extractor._get_transformer_layers())} layers")
    
    def load_training_data(self, target_operator: str, data_dir: str = "probe_data") -> List[Dict]:
        op_clean = target_operator.replace("*", "mult")
        filepath = os.path.join(data_dir, f"training_data_{op_clean}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training data not found: {filepath}. Please run data_preprocessing.py first.")
        
        with open(filepath, 'rb') as f:
            training_data = pickle.load(f)
        
        print(f"Loaded {len(training_data)} training samples for operator '{target_operator}'")
        return training_data
    
    def extract_representations_from_data(self, training_data: List[Dict], 
                                        layers_to_analyze: List[int] = None) -> Dict:
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.model_config["num_layers"])) 
        
        all_prompts = [item['prompt'] for item in training_data]
        all_labels = [item['label'] for item in training_data]
        
        representations = self.extractor.extract_representations(
            all_prompts, layers_to_analyze, return_attention=False
        )
        
        layer_representations = {}
        for layer_idx in layers_to_analyze:
            layer_name = f"layer_{layer_idx}"
            layer_reprs = []
            
            for sample_idx in range(len(all_prompts)):
                sample_key = f"sample_{sample_idx}"
                if sample_key in representations and layer_name in representations[sample_key]['hidden_states']:
                    # Extract final token representation
                    hidden_state = representations[sample_key]['hidden_states'][layer_name]
                    final_token_repr = hidden_state[0, -1, :].detach().cpu().float().numpy()
                    layer_reprs.append(final_token_repr)
                else:
                    print(f"Warning: Missing representation for sample {sample_idx}, layer {layer_idx}")
                    hidden_dim = self.model_config["hidden_dim"]
                    layer_reprs.append(np.zeros(hidden_dim))
            
            layer_representations[layer_idx] = {
                'representations': np.array(layer_reprs),
                'labels': np.array(all_labels)
            }
        
        return layer_representations
    
    def train_probe_for_layer(self, representations: np.ndarray, labels: np.ndarray,
                             layer_idx: int, target_operator: str,
                             test_size: float = 0.2, epochs: int = 100,
                             batch_size: int = 32, learning_rate: float = 1e-3,
                             weight_decay: float = 1e-4) -> Dict:

        X_train, X_test, y_train, y_test = train_test_split(
            representations, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = ProbeDataset(X_train, y_train)
        test_dataset = ProbeDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = representations.shape[1]
        probe = LinearProbe(input_dim).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        probe.train()
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = probe(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"Layer {layer_idx} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if patience_counter >= early_stopping_patience and epoch > 20:
                print(f"Layer {layer_idx} - Early stopping at epoch {epoch+1}")
                break
        
        # Evaluation
        probe.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = probe(batch_X)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.detach().cpu().numpy())
                all_true.extend(batch_y.detach().cpu().numpy())
        
        accuracy = accuracy_score(all_true, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='binary')
        
        return {
            'probe_state_dict': probe.cpu().state_dict(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_losses': train_losses,
            'layer_idx': layer_idx,
            'target_operator': target_operator,
            'input_dim': input_dim
        }
    
    def train_probes_all_layers(self, target_operator: str,
                               layers_to_analyze: List[int] = None,
                               save_dir: str = "trained_probes") -> Dict:
        
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.model_config["num_layers"]))
        
        training_data = self.load_training_data(target_operator)
        layer_representations = self.extract_representations_from_data(training_data, layers_to_analyze)

        trained_probes = {}
        print(f"\nTraining linear probes for operator '{target_operator}' across {len(layers_to_analyze)} layers...")
        
        for layer_idx in tqdm(layers_to_analyze, desc="Training probes"):
            layer_data = layer_representations[layer_idx]
            
            probe_results = self.train_probe_for_layer(
                layer_data['representations'],
                layer_data['labels'],
                layer_idx,
                target_operator
            )
            
            trained_probes[layer_idx] = probe_results
            
            acc = probe_results['accuracy']
            f1 = probe_results['f1']
            
            if layer_idx % 3 == 0:
                print(f"Layer {layer_idx}: Accuracy={acc:.3f}, F1={f1:.3f}")
        
        os.makedirs(save_dir, exist_ok=True)
        model_clean = self.model_name.replace("/", "_").replace("-", "_")
        op_clean = target_operator.replace("*", "mult")
        save_path = os.path.join(save_dir, f"probes_{model_clean}_{op_clean}.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(trained_probes, f)
        
        return trained_probes

def main():
    
    
    parser = argparse.ArgumentParser(description='Train Linear Probes')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B",
                        help='Model name')
    parser.add_argument('--model_type', type=str, default="qwen3",
                        help='Model type')
    parser.add_argument('--operator', type=str, default='+',
                        help='Target operator (+, -, *)')
    
    args = parser.parse_args()
    
    trainer = LinearProbeTrainer(args.model_name, args.model_type)
    trainer.load_model()
    trained_probes = trainer.train_probes_all_layers(target_operator=args.operator)

if __name__ == "__main__":
    main()