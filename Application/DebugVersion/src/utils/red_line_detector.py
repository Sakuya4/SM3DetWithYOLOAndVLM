"""
紅線偵測模組
負責偵測影像中的紅線區域
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_red_lines(img: np.ndarray) -> np.ndarray:
    """
    偵測影像中的紅線 - 精確版本，專注於道路底部的水平紅線
    
    Args:
        img: 輸入影像 (BGR 格式)
    
    Returns:
        red_mask: 紅線遮罩 (二值化影像)
    """
    try:
        # 轉換到 HSV 色彩空間
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 更精確的紅色範圍 - 專注於道路紅線
        # 道路紅線通常有較高的飽和度和適中的亮度
        lower_red1 = np.array([0, 100, 50])       # 低紅色 - 更高飽和度要求
        upper_red1 = np.array([10, 255, 255])    # 高紅色 - 更嚴格的上限
        
        
        lower_red2 = np.array([170, 100, 50])    # 低紅色 (接近 180 度)
        upper_red2 = np.array([180, 255, 255])    # 高紅色 (接近 180 度)
        
        # 創建兩個紅色遮罩
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 合併兩個遮罩
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 多階段形態學操作
        # 1. 先開運算去除小雜訊
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 2. 水平方向連接斷線 (道路紅線通常是水平的)
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horizontal, iterations=1)
        
        # 3. 垂直方向稍微連接 (允許紅線有輕微的垂直變化)
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical, iterations=1)
        
        # 4. 最終清理小雜訊
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 過濾：只保留符合紅線特徵的區域
        # mask = filter_red_line_features(mask, img.shape)
        
        # 最終優化：移除孤立的像素點
        # mask = remove_isolated_pixels(mask)
        
        logger.info("紅線偵測完成 (精確版本)")
        return mask
        
    except Exception as e:
        logger.error(f"紅線偵測失敗: {e}")
        # 返回空遮罩
        return np.zeros(img.shape[:2], dtype=np.uint8)

def bbox_hits_redline(bbox: tuple, red_mask: np.ndarray, 
                      band_px: int = 50, min_ratio: float = 0.003) -> tuple:
    """
    檢查車輛邊界框是否與紅線重疊
    
    Args:
        bbox: 邊界框座標 (x1, y1, x2, y2)
        red_mask: 紅線遮罩
        band_px: 檢查區域高度 (像素)
        min_ratio: 最小重疊比例
    
    Returns:
        (overlapped, ratio): (是否重疊, 重疊比例)
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # 確保座標在影像範圍內
        h, w = red_mask.shape
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 檢查底部區域 (假設紅線在底部)
        y_top = max(y2 - band_px, 0)
        strip = red_mask[y_top:y2+1, x1:x2+1]
        
        if strip.size == 0:
            return False, 0.0
        
        # 計算紅線像素比例
        red_pixels = cv2.countNonZero(strip)
        ratio = red_pixels / strip.size
        
        overlapped = ratio >= min_ratio
        
        logger.debug(f"紅線重疊檢查: bbox={bbox}, ratio={ratio:.3f}, overlapped={overlapped}")
        
        return overlapped, ratio
        
    except Exception as e:
        logger.error(f"紅線重疊檢查失敗: {e}")
        return False, 0.0

def enhance_red_line_detection(img: np.ndarray, 
                              brightness_factor: float = 1.5, 
                              contrast_factor: float = -0.5) -> np.ndarray:
    """
    增強影像以改善紅線偵測
    
    Args:
        img: 輸入影像
        brightness_factor: 亮度增強因子
        contrast_factor: 對比度增強因子
    
    Returns:
        enhanced_img: 增強後的影像
    """
    try:
        # 高斯模糊
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        
        # 亮度與對比度調整
        enhanced = cv2.addWeighted(img, brightness_factor, blurred, contrast_factor, 0)
        
        logger.info("影像增強完成")
        return enhanced
        
    except Exception as e:
        logger.error(f"影像增強失敗: {e}")
        return img

def filter_red_line_features(mask: np.ndarray, img_shape: tuple) -> np.ndarray:
    """
    過濾紅線特徵，只保留符合道路紅線特徵的區域
    
    Args:
        mask: 原始紅色遮罩
        img_shape: 影像尺寸 (height, width)
    
    Returns:
        filtered_mask: 過濾後的紅線遮罩
    """
    try:
        h, w = img_shape[:2]
        
        # 找到所有輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 創建新的遮罩
        filtered_mask = np.zeros_like(mask)
        
        # 道路紅線的優先級排序
        road_line_candidates = []
        
        for contour in contours:
            # 計算輪廓面積
            area = cv2.contourArea(contour)
            
            # 過濾條件 1: 面積必須在合理範圍內
            min_area = 100   # 提高最小面積要求，避免雜訊
            max_area = (h * w) // 20  # 降低最大面積限制，更嚴格
            
            if area < min_area or area > max_area:
                continue
            
            # 過濾條件 2: 長寬比必須符合紅線特徵
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1)
            
            # 道路紅線通常是長條形，長寬比應該 > 4 (更嚴格)
            if aspect_ratio < 4.0:
                continue
            
            # 過濾條件 3: 檢查是否為水平線 (道路紅線通常是水平的)
            if len(contour) >= 5:
                try:
                    # 擬合橢圓
                    ellipse = cv2.fitEllipse(contour)
                    angle = ellipse[2]  # 橢圓角度
                    
                    # 角度應該接近 0 度 (水平線)
                    # 允許 ±15 度的誤差 (更嚴格)
                    angle_normalized = angle % 90
                    if angle_normalized > 15 and angle_normalized < 75:
                        continue
                except:
                    # 如果橢圓擬合失敗，跳過
                    pass
            
            # 過濾條件 4: 檢查位置合理性
            center_y = y + h_rect // 2
            center_x = x + w_rect // 2
            
            # 道路紅線必須在影像的底部區域
            bottom_region = h * 0.8  # 底部 20% 區域 (更嚴格)
            
            # 計算位置分數 (越靠近底部分數越高)
            position_score = 0
            if center_y >= bottom_region:
                position_score = 100  # 底部區域
            elif center_y >= h * 0.6:
                position_score = 50   # 中下部區域
            else:
                position_score = 0    # 上部區域
            
            # 過濾條件 5: 檢查形狀合理性
            # 道路紅線不應該是圓形或不規則形狀
            if w_rect > h_rect:  # 水平紅線
                if w_rect > 300 or h_rect > 50:  # 寬度不超過300，高度不超過50
                    continue
            else:  # 垂直紅線 (較少見)
                if h_rect > 300 or w_rect > 50:  # 高度不超過300，寬度不超過50
                    continue
            
            # 計算綜合分數
            total_score = position_score + (aspect_ratio * 10) + (area / 100)
            
            road_line_candidates.append({
                'contour': contour,
                'score': total_score,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'position': center_y,
                'bbox': (x, y, w_rect, h_rect)
            })
        
        # 按分數排序，只保留最高分的候選者
        road_line_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 只保留前 3 個最佳候選者
        for candidate in road_line_candidates[:3]:
            cv2.fillPoly(filtered_mask, [candidate['contour']], 255)
            logger.debug(f"保留紅線候選者: 分數={candidate['score']:.1f}, 面積={candidate['area']:.0f}, 長寬比={candidate['aspect_ratio']:.1f}")
        
        logger.info(f"紅線特徵過濾完成: 原始輪廓數={len(contours)}, 過濾後輪廓數={len(road_line_candidates[:3])}")
        return filtered_mask
        
    except Exception as e:
        logger.error(f"紅線特徵過濾失敗: {e}")
        return mask

def remove_isolated_pixels(mask: np.ndarray) -> np.ndarray:
    """
    移除孤立的像素點，保留連續的紅線區域
    
    Args:
        mask: 輸入遮罩
    
    Returns:
        cleaned_mask: 清理後的遮罩
    """
    try:
        # 使用連通元件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 創建新的遮罩
        cleaned_mask = np.zeros_like(mask)
        
        # 只保留面積足夠大的連通元件
        min_component_area = 200  # 最小連通元件面積
        
        for i in range(1, num_labels):  # 跳過背景 (標籤 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                # 提取這個連通元件
                component_mask = (labels == i).astype(np.uint8) * 255
                cleaned_mask = cv2.bitwise_or(cleaned_mask, component_mask)
        
        return cleaned_mask
        
    except Exception as e:
        logger.error(f"移除孤立像素失敗: {e}")
        return mask

def get_red_line_statistics(red_mask: np.ndarray) -> dict:
    """
    獲取紅線統計資訊
    
    Args:
        red_mask: 紅線遮罩
    
    Returns:
        stats: 統計資訊字典
    """
    try:
        total_pixels = red_mask.size
        red_pixels = cv2.countNonZero(red_mask)
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
        
        # 計算紅線區域的輪廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 計算最大紅線區域面積
        max_area = 0
        if contours:
            max_area = max(cv2.contourArea(contour) for contour in contours)
        
        stats = {
            'total_pixels': total_pixels,
            'red_pixels': red_pixels,
            'red_ratio': red_ratio,
            'contour_count': len(contours),
            'max_red_area': max_area
        }
        
        logger.info(f"紅線統計: 紅線比例={red_ratio:.3f}, 輪廓數={len(contours)}")
        return stats
        
    except Exception as e:
        logger.error(f"紅線統計計算失敗: {e}")
        return {}
