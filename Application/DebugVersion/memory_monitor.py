import psutil
import torch
import gc
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, process_id: Optional[int] = None):
        self.process_id = process_id or psutil.Process().pid
        self.process = psutil.Process(self.process_id)
        self.baseline_memory = 0
        self.memory_history: List[Dict] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.leak_threshold = 1.2
        self.max_history = 100 
        
    def start_monitoring(self, interval: float = 1.0):
        if self.monitoring:
            logger.warning("記憶體監控已在運行中")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        logger.info(f"開始記憶體監控，進程 ID: {self.process_id}")
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("停止記憶體監控")
        
    def _monitor_loop(self, interval: float):
        while self.monitoring:
            try:
                self.take_snapshot()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"監控過程中發生錯誤: {e}")
                
    def take_snapshot(self):
        try:
            system_memory = psutil.virtual_memory()
            
            process_memory = self.process.memory_info()
            
            gpu_memory = self._get_gpu_memory()
            
            snapshot = {
                'timestamp': datetime.now(),
                'system_total': system_memory.total / (1024**3),  # GB
                'system_available': system_memory.available / (1024**3),  # GB
                'system_percent': system_memory.percent,
                'process_rss': process_memory.rss / (1024**3),  # GB
                'process_vms': process_memory.vms / (1024**3),  # GB
                'gpu_allocated': gpu_memory.get('allocated', 0) / (1024**3),  # GB
                'gpu_reserved': gpu_memory.get('reserved', 0) / (1024**3),  # GB
            }
            
            self.memory_history.append(snapshot)
            
            if len(self.memory_history) > self.max_history:
                self.memory_history.pop(0)
                
            if self.baseline_memory == 0:
                self.baseline_memory = snapshot['process_rss']
                logger.info(f"設定基準記憶體: {self.baseline_memory:.2f} GB")
                
        except Exception as e:
            logger.error(f"拍攝記憶體快照失敗: {e}")
            
    def _get_gpu_memory(self) -> Dict[str, int]:
        try:
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved(),
                    'total': torch.cuda.get_device_properties(0).total_memory
                }
        except Exception as e:
            logger.warning(f"無法獲取 GPU 記憶體資訊: {e}")
        return {'allocated': 0, 'reserved': 0, 'total': 0}
        
    def detect_memory_leak(self) -> Dict[str, any]:
        if len(self.memory_history) < 5:
            return {'leak_detected': False, 'reason': '歷史記錄不足'}
            
        recent_snapshots = self.memory_history[-5:]
        avg_memory = sum(s['process_rss'] for s in recent_snapshots) / len(recent_snapshots)
        
        growth_rate = avg_memory / self.baseline_memory if self.baseline_memory > 0 else 1.0
        
        leak_detected = growth_rate > self.leak_threshold
        
        return {
            'leak_detected': leak_detected,
            'baseline_memory': self.baseline_memory,
            'current_average': avg_memory,
            'growth_rate': growth_rate,
            'threshold': self.leak_threshold,
            'severity': 'high' if growth_rate > 2.0 else 'medium' if growth_rate > 1.5 else 'low'
        }
        
    def get_memory_stats(self) -> Dict[str, any]:
        if not self.memory_history:
            return {}
            
        latest = self.memory_history[-1]
        
        if len(self.memory_history) >= 10:
            recent_avg = sum(s['process_rss'] for s in self.memory_history[-10:]) / 10
            older_avg = sum(s['process_rss'] for s in self.memory_history[-20:-10]) / 10
            trend = 'increasing' if recent_avg > older_avg else 'decreasing' if recent_avg < older_avg else 'stable'
        else:
            trend = 'insufficient_data'
            
        return {
            'current_memory': latest['process_rss'],
            'system_memory_percent': latest['system_percent'],
            'gpu_memory_allocated': latest['gpu_allocated'],
            'gpu_memory_reserved': latest['gpu_reserved'],
            'memory_trend': trend,
            'history_count': len(self.memory_history)
        }
        
    def generate_report(self) -> str:
        if not self.memory_history:
            return "沒有記憶體使用數據"
            
        latest = self.memory_history[-1]
        leak_info = self.detect_memory_leak()
        stats = self.get_memory_stats()
        
        report = f"""
=== 記憶體使用報告 ===
時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
進程 ID: {self.process_id}

系統記憶體:
  總量: {latest['system_total']:.2f} GB
  可用: {latest['system_available']:.2f} GB
  使用率: {latest['system_percent']:.1f}%

進程記憶體:
  當前使用: {latest['process_rss']:.2f} GB
  虛擬記憶體: {latest['process_vms']:.2f} GB

GPU 記憶體:
  已分配: {latest['gpu_allocated']:.2f} GB
  已保留: {latest['gpu_reserved']:.2f} GB

記憶體洩漏檢測:
  是否洩漏: {'是' if leak_info['leak_detected'] else '否'}
  基準記憶體: {leak_info['baseline_memory']:.2f} GB
  當前平均: {leak_info['current_average']:.2f} GB
  增長率: {leak_info['growth_rate']:.2f}x
  嚴重程度: {leak_info['severity']}

記憶體趨勢: {stats['memory_trend']}
歷史記錄數: {stats['history_count']}
"""
        return report
        
    def clear_memory(self):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            logger.info("記憶體清理完成")
            
        except Exception as e:
            logger.error(f"記憶體清理失敗: {e}")
            
    def export_history(self, filename: str = None):
        if not filename:
            filename = f"memory_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("時間戳,系統總量(GB),系統可用(GB),系統使用率(%),進程RSS(GB),進程VMS(GB),GPU已分配(GB),GPU已保留(GB)\n")
                for snapshot in self.memory_history:
                    f.write(f"{snapshot['timestamp']},{snapshot['system_total']:.3f},{snapshot['system_available']:.3f},{snapshot['system_percent']:.1f},{snapshot['process_rss']:.3f},{snapshot['process_vms']:.3f},{snapshot['gpu_allocated']:.3f},{snapshot['gpu_reserved']:.3f}\n")
            logger.info(f"記憶體歷史已匯出到: {filename}")
        except Exception as e:
            logger.error(f"匯出記憶體歷史失敗: {e}")

_global_monitor: Optional[MemoryMonitor] = None

def get_global_monitor() -> MemoryMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor

def start_global_monitoring(interval: float = 1.0):
    monitor = get_global_monitor()
    monitor.start_monitoring(interval)

def stop_global_monitoring():
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()

def get_memory_info() -> Dict[str, any]:
    monitor = get_global_monitor()
    return monitor.get_memory_stats()

def check_memory_leak() -> Dict[str, any]:
    monitor = get_global_monitor()
    return monitor.detect_memory_leak()

def print_memory_report():
    monitor = get_global_monitor()
    print(monitor.generate_report())

def clear_memory():
    monitor = get_global_monitor()
    monitor.clear_memory()
