"""
ENHANCED SLM TRAINING & DEPLOYMENT - PRODUCTION READY
======================================================

Features:
1. Continuous fine-tuning (load checkpoints, resume training)
2. Custom dataset loading
3. Extended API endpoints for AI agent integration
4. Detailed training metrics & statistics
5. Model versioning & checkpoint management

Model: SmolLM2-135M-Instruct (~540MB Memory Footprint)
Device: CPU (Safe for Low-RAM Environments)
"""

import sys
try:
    import _lzma
except ImportError:
    from backports import lzma
    sys.modules['_lzma'] = lzma

import os
import json
import logging
import shutil
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)
from flask import Flask, request, jsonify
from datetime import datetime
import gc
import warnings
import pickle

# Suppress non-critical warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class OptimizedConfig:
    # SmolLM2 135M (~540MB base size)
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    MODEL_MAX_LENGTH = 256
    
    # Training parameters
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 2  # Increased from 1 for better fine-tuning
    
    # Paths & Hardware
    DATA_DIR = "./data"
    OUTPUT_DIR = "./models/smollm_finetuned"
    CHECKPOINT_DIR = "./checkpoints"
    DEVICE = "cpu"
    LOG_LEVEL = logging.INFO
    
    # New: Stats tracking
    STATS_FILE = "./stats/training_stats.json"
    METRICS_FILE = "./stats/metrics.json"

logging.basicConfig(
    level=OptimizedConfig.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slm_workflow_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STATISTICS TRACKER
# ============================================================================

class TrainingStats:
    """Track training metrics for monitoring and analysis"""
    
    def __init__(self, stats_file: str):
        self.stats_file = stats_file
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict:
        """Load existing stats or create new"""
        os.makedirs(os.path.dirname(self.stats_file) or '.', exist_ok=True)
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {
            'total_epochs': 0,
            'total_batches': 0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
            'training_start': datetime.now().isoformat(),
            'last_checkpoint': None,
            'num_checkpoints': 0,
            'inference_count': 0,
            'avg_inference_time': 0.0
        }
    
    def update_batch(self, loss: float):
        """Update after processing a batch"""
        self.stats['total_batches'] += 1
        self.stats['total_loss'] += loss
        self.stats['avg_loss'] = self.stats['total_loss'] / self.stats['total_batches']
    
    def update_epoch(self):
        """Update after epoch completion"""
        self.stats['total_epochs'] += 1
    
    def update_checkpoint(self, path: str):
        """Update checkpoint info"""
        self.stats['last_checkpoint'] = path
        self.stats['num_checkpoints'] += 1
        self.stats['last_checkpoint_time'] = datetime.now().isoformat()
    
    def update_inference(self, time_taken: float):
        """Update inference metrics"""
        self.stats['inference_count'] += 1
        if self.stats['avg_inference_time'] == 0:
            self.stats['avg_inference_time'] = time_taken
        else:
            # Running average
            n = self.stats['inference_count']
            self.stats['avg_inference_time'] = (
                (self.stats['avg_inference_time'] * (n - 1) + time_taken) / n
            )
    
    def save(self):
        """Persist stats to file"""
        self.stats['last_updated'] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.stats_file) or '.', exist_ok=True)
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Stats saved: {self.stats_file}")
    
    def get_summary(self) -> Dict:
        """Get summary for API"""
        return self.stats.copy()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_stale_directories(dirs: List[str]):
    for d in dirs:
        if os.path.exists(d):
            logger.info(f"Cleaning stale folder: {d}")
            try:
                shutil.rmtree(d)
            except Exception as e:
                logger.warning(f"Failed to clear {d}: {str(e)}")

def create_sample_training_data(output_file: str):
    """Create diverse neurofeedback training data"""
    training_data = [
        # EEG-based interventions
        {"input": "EEG: Alpha high", "output": "Relaxation state detected. Continue current activity."},
        {"input": "EEG: Beta spike", "output": "Stress detected. Try box breathing: 4 count in, hold 4, out 4."},
        {"input": "EEG: Theta prominent", "output": "Drowsiness detected. Take a break, stretch, or drink water."},
        {"input": "EEG: Gamma high", "output": "Good focus state. Continue deep work session."},
        {"input": "EEG: Delta waves", "output": "Recovery mode detected. Maintain rest, sleep is restorative."},
        
        # Stress-based interventions
        {"input": "Stress: High, State: CRITICAL", "output": "Immediate intervention needed. Practice box breathing or 5-min guided meditation."},
        {"input": "Stress: Moderate, State: BASELINE", "output": "Stable state. Maintain current activity or take light walk."},
        {"input": "Stress: Low, State: CALM", "output": "Optimal relaxation. Good time for meditation or creative work."},
        
        # Heart rate interventions
        {"input": "Heart rate: 95 BPM, Stress: 75", "output": "Elevated heart rate suggests stress. Try slow deep breathing."},
        {"input": "Heart rate: 58 BPM, Stress: 30", "output": "Resting heart rate optimal. Continue recovery mode."},
        
        # Combined signals
        {"input": "Alpha: 0.4, Beta: 0.6, HR: 88, Stress: 70", "output": "Mixed stress signals. Prioritize relaxation technique over work."},
        {"input": "Gamma: 0.8, Alpha: 0.6, HR: 70, Stress: 50", "output": "Flow state indicators present. Maintain focus on current task."},
    ]
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"Created sample training data with {len(training_data)} examples: {output_file}")
    return output_file

# ============================================================================
# DATASET
# ============================================================================

class NeurofeedbackDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self._load_data(file_path)
    
    def _load_data(self, file_path: str):
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            try:
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                full_text = f"Input: {input_text}\nOutput: {output_text}"
                
                encodings = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].squeeze(0)
                attention_mask = encodings['attention_mask'].squeeze(0)
                
                # Mask out padding tokens in labels
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                self.examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })
            except Exception as e:
                logger.warning(f"Failed to parse item {idx}: {e}")
                continue
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# ============================================================================
# TRAINER WITH CHECKPOINT SUPPORT
# ============================================================================

class OptimizedSLMTrainer:
    def __init__(self, config: OptimizedConfig, stats: TrainingStats):
        self.config = config
        self.stats = stats
        self.tokenizer = None
        self.model = None
        self.device = config.DEVICE
    
    def load_model_and_tokenizer(self, checkpoint_path: str = None):
        """Load model, optionally from checkpoint for continuous training"""
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"[LOADING] Resuming from checkpoint: {checkpoint_path}")
            load_path = checkpoint_path
        else:
            logger.info("[LOADING] Starting from base model")
            load_path = self.config.MODEL_NAME
        
        logger.info("[STEP 1] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            padding_side='right'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("[STEP 2] Loading model configuration...")
        model_config = AutoConfig.from_pretrained(load_path)
        
        logger.info("[STEP 3] Loading model with eager attention (GQA fix)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            config=model_config,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info("✓ Model initialized successfully")
    
    def prepare_training_data(self, data_file: str):
        #if not os.path.exists(data_file):
        #    create_sample_training_data(data_file)
        
        dataset = NeurofeedbackDataset(
            file_path=data_file,
            tokenizer=self.tokenizer,
            max_length=self.config.MODEL_MAX_LENGTH
        )
        
        train_size = max(1, int(0.8 * len(dataset)))  # 80/20 split
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        return DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True), \
               DataLoader(val_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
    
    def fine_tune(self, train_loader: DataLoader, val_loader: DataLoader):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            optimizer.zero_grad()
            successful_batches = 0
            total_batches = 0
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                total_batches += 1
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    if loss is None or torch.isnan(loss):
                        continue
                    
                    loss_scaled = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                    loss_scaled.backward()
                    
                    if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        gc.collect()
                    
                    epoch_loss += loss.item()
                    self.stats.update_batch(loss.item())
                    
                    logger.info(f"Epoch {epoch + 1} | Batch {batch_idx} Loss: {loss.item():.4f}")
                    successful_batches += 1
                        
                except Exception as e:
                    logger.error(f"Batch {batch_idx} error: {str(e)}")
            
            if successful_batches == 0 and total_batches > 0:
                raise RuntimeError(
                    f"Epoch {epoch + 1}: all {total_batches} batches failed - "
                    "training did not actually happen. Refusing to save."
                )
            
            avg_epoch_loss = epoch_loss / successful_batches if successful_batches > 0 else 0
            self.stats.update_epoch()
            
            logger.info(
                f"Epoch {epoch + 1} completed: {successful_batches}/{total_batches} batches succeeded. "
                f"Avg Loss: {avg_epoch_loss:.4f}"
            )
    
    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.stats.update_checkpoint(output_dir)
        self.stats.save()
        logger.info(f"✓ Model saved to {output_dir}")

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class OptimizedInferenceEngine:
    def __init__(self, model_dir: str, device: str = 'cpu', stats: TrainingStats = None):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.stats = stats
        self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        logger.info("✓ Inference model loaded successfully")
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        start_time = datetime.now()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        gen_time = (datetime.now() - start_time).total_seconds()
        gen_tokens = len(outputs[0]) - len(input_ids[0])
        
        if self.stats:
            self.stats.update_inference(gen_time)
        
        return {
            'prompt': prompt,
            'generated_text': generated_text.strip(),
            'num_tokens': gen_tokens,
            'generation_time': gen_time,
            'tokens_per_second': gen_tokens / gen_time if gen_time > 0 else 0
        }
    
    def analyze_eeg(self, eeg_dict: Dict) -> Dict:
        """Analyze EEG data and provide intervention"""
        
        stress_level = eeg_dict.get('stress_level', 50)
        brain_state = eeg_dict.get('brain_state', 'BASELINE')
        
        # Create EEG-aware prompt
        prompt = (
            f"EEG Analysis: State={brain_state}, Stress={stress_level}/100, "
            f"Alpha={eeg_dict.get('alpha_power', 0.3)}, "
            f"Beta={eeg_dict.get('beta_power', 0.4)}, "
            f"HR={eeg_dict.get('heart_rate', 70)} BPM. "
            f"Recommendation:"
        )
        
        result = self.generate_response(prompt, max_new_tokens=60)
        return {
            'eeg_input': eeg_dict,
            'analysis': result['generated_text'],
            'generation_time': result['generation_time']
        }

# ============================================================================
# ENHANCED API SERVER
# ============================================================================

def create_api_server(inference_engine: OptimizedInferenceEngine, stats: TrainingStats):
    app = Flask(__name__)
    app.config['inference_engine'] = inference_engine
    app.config['stats'] = stats
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'model': 'SmolLM2-135M-Instruct',
            'device': 'CPU',
            'optimization': 'GQA + Eager Attention'
        }), 200
    
    @app.route('/api/v1/generate', methods=['POST'])
    def generate():
        """Generate text from prompt"""
        try:
            data = request.get_json() or {}
            prompt = data.get('prompt', '')
            if not prompt:
                return jsonify({'error': 'Prompt required'}), 400
            
            result = app.config['inference_engine'].generate_response(
                prompt=prompt,
                max_new_tokens=data.get('max_tokens', 50),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.9)
            )
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v1/analyze_eeg', methods=['POST'])
    def analyze_eeg():
        """Analyze EEG data and provide intervention"""
        try:
            eeg_data = request.get_json() or {}
            
            result = app.config['inference_engine'].analyze_eeg(eeg_data)
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"EEG analysis error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v1/batch_generate', methods=['POST'])
    def batch_generate():
        """Generate text for multiple prompts"""
        try:
            data = request.get_json() or {}
            prompts = data.get('prompts', [])
            
            if not prompts:
                return jsonify({'error': 'Prompts array required'}), 400
            
            results = []
            for prompt in prompts:
                result = app.config['inference_engine'].generate_response(
                    prompt=prompt,
                    max_new_tokens=data.get('max_tokens', 50)
                )
                results.append(result)
            
            return jsonify({'results': results, 'count': len(results)}), 200
        except Exception as e:
            logger.error(f"Batch generation error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v1/stats', methods=['GET'])
    def get_stats():
        """Get training and inference statistics"""
        try:
            stats_summary = app.config['stats'].get_summary()
            return jsonify(stats_summary), 200
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v1/model_info', methods=['GET'])
    def model_info():
        """Get model information"""
        try:
            engine = app.config['inference_engine']
            return jsonify({
                'model_name': 'SmolLM2-135M-Instruct',
                'model_size_mb': '540',
                'max_length': '256',
                'device': 'CPU',
                'dtype': 'float32',
                'attention_type': 'Grouped Query Attention (GQA)',
                'checkpoint_path': engine.model_dir
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
        'service': 'Manah SLM API',
        'endpoints': ['/health', '/api/v1/generate', '/api/v1/analyze_eeg',
                      '/api/v1/batch_generate', '/api/v1/stats', '/api/v1/model_info']
        }), 200

    @app.route('/favicon.ico')
    def favicon():
        return '', 204
    
    return app

# ============================================================================
# WORKFLOW ORCHESTRATOR
# ============================================================================

class OptimizedWorkflow:
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.stats = TrainingStats(config.STATS_FILE)
    
    def run_complete_workflow(self, data_file: str = None, resume_checkpoint: str = None):
        """Run training and launch API server"""
        
        logger.info("=" * 80)
        logger.info("STARTING ENHANCED SLM WORKFLOW")
        logger.info("=" * 80)
        
        # Cleanup (optional - comment out to keep existing checkpoints)
        # clean_stale_directories([self.config.OUTPUT_DIR])
        
        if data_file is None:
            data_file = os.path.join(self.config.DATA_DIR, 'training_data.json')
            create_sample_training_data(data_file)
        
        # Training
        trainer = OptimizedSLMTrainer(self.config, self.stats)
        trainer.load_model_and_tokenizer(checkpoint_path=resume_checkpoint)
        
        train_loader, val_loader = trainer.prepare_training_data(data_file)
        trainer.fine_tune(train_loader, val_loader)
        trainer.save_model(self.config.OUTPUT_DIR)
        
        del trainer
        gc.collect()
        
        # Inference server
        logger.info("\n" + "=" * 80)
        logger.info("STARTING INFERENCE SERVER")
        logger.info("=" * 80)
        
        engine = OptimizedInferenceEngine(
            model_dir=self.config.OUTPUT_DIR,
            device=self.config.DEVICE,
            stats=self.stats
        )
        return create_api_server(engine, self.stats)

if __name__ == "__main__":
    config = OptimizedConfig()
    workflow = OptimizedWorkflow(config)
    
    try:
        app = workflow.run_complete_workflow()
        logger.info("✓ API Server launching on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        raise
