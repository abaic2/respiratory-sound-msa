import cv2
import numpy as np
import os
from PIL import Image

def reshape_image_to_spectrogram_size(image, target_height=1024, target_width=512):
    """
    将图片重塑为频谱图尺寸 (高度1024, 宽度512)
    
    Args:
        image: 输入图片
        target_height: 目标高度，默认1024
        target_width: 目标宽度，默认512
    
    Returns:
        resized_image: 重塑后的图片
    """
    height, width = image.shape[:2]
    print(f"📏 原始图片尺寸: 宽度{width} × 高度{height}")
    
    # 使用高质量的双三次插值进行缩放
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    print(f"🔄 重塑后尺寸: 宽度{target_width} × 高度{target_height}")
    
    return resized_image

def split_image_into_patches(image_path, output_folder, num_patches=8, reshape_to_spectrogram=True):
    """
    将图片重塑为(高度1024,宽度512)然后分成指定数量的等大小正方形patch
    
    Args:
        image_path: 输入图片路径
        output_folder: 输出文件夹路径
        num_patches: patch数量，默认8个
        reshape_to_spectrogram: 是否重塑为频谱图尺寸，默认True
    """
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取图片
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    # 🔄 步骤1：重塑图片为频谱图尺寸（如果需要）
    if reshape_to_spectrogram:
        print("\n🔄 步骤1：重塑图片为频谱图尺寸 (高度1024×宽度512)")
        image = reshape_image_to_spectrogram_size(original_image, 1024, 512)
        
        # 保存重塑后的图片
        resized_path = os.path.join(output_folder, "resized_1024x512.png")
        cv2.imwrite(resized_path, image)
        print(f"💾 重塑后的图片保存至: {resized_path}")
    else:
        image = original_image
        print(f"📏 使用原图尺寸进行分割")
    
    # 获取当前图片尺寸
    height, width = image.shape[:2]  # height=1024, width=512
    print(f"\n📐 当前图片尺寸: 宽度{width} × 高度{height}")
    
    # 📊 步骤2：计算最佳patch布局
    print(f"\n📊 步骤2：计算patch分割布局")
    
    # 对于(高度1024, 宽度512)的频谱图，使用特殊的布局策略
    if reshape_to_spectrogram and height == 1024 and width == 512:
        print("🎯 检测到频谱图尺寸 (1024×512)，使用优化布局策略")
        
        # 对于1024×512，最佳布局是4行2列
        rows, cols = 4, 2
        
        # 计算patch尺寸：使其尽可能接近正方形
        # 网格尺寸：高度256(1024/4)，宽度256(512/2)
        grid_height = height // rows  # 256
        grid_width = width // cols    # 256
        
        # patch尺寸为正方形
        patch_size = min(grid_height, grid_width)  # 256
        
        print(f"📏 网格尺寸: 宽度{grid_width} × 高度{grid_height}")
        print(f"🔲 Patch尺寸: {patch_size} × {patch_size} (正方形)")
        print(f"📐 布局: {rows}行 × {cols}列")
    else:
        # 标准布局计算
        rows = 4
        cols = 2
        
        # 计算每个patch的尺寸
        patch_height = height // rows
        patch_width = width // cols
        
        # 确保patch是正方形 - 取较小的尺寸
        patch_size = min(patch_height, patch_width)
        
        print(f"📏 标准布局: {rows}行 × {cols}列")
        print(f"🔲 Patch尺寸: {patch_size} × {patch_size}")
    
    # 🔨 步骤3：分割并保存patch
    print(f"\n🔨 步骤3：分割并保存patch")
    
    patch_count = 0
    patch_info = []
    
    for row in range(rows):
        for col in range(cols):
            if patch_count >= num_patches:
                break
            
            if reshape_to_spectrogram and height == 1024 and width == 512:
                # 频谱图优化分割：4行2列网格分布
                # 每个patch占用256×256像素
                y1 = row * (height // rows)  # row * 256 (0, 256, 512, 768)
                y2 = y1 + patch_size         # y1 + 256
                x1 = col * (width // cols)   # col * 256 (0, 256)
                x2 = x1 + patch_size         # x1 + 256
                
                print(f"🎯 Patch {patch_count + 1}: 第{row+1}行第{col+1}列 → 坐标[x:{x1}-{x2}, y:{y1}-{y2}]")
            else:
                # 标准网格分割
                y1 = row * patch_size
                x1 = col * patch_size
                x2 = x1 + patch_size
                y2 = y1 + patch_size
            
            # 提取patch
            patch = image[y1:y2, x1:x2]
            
            # 验证patch尺寸
            actual_height, actual_width = patch.shape[:2]
            if actual_height != patch_size or actual_width != patch_size:
                print(f"⚠️ Patch {patch_count + 1} 尺寸异常: 宽度{actual_width}×高度{actual_height}, 预期: {patch_size}×{patch_size}")
            
            # 保存patch
            patch_filename = f"patch_{patch_count + 1:02d}.png"
            patch_path = os.path.join(output_folder, patch_filename)
            
            success = cv2.imwrite(patch_path, patch)
            if success:
                print(f"✅ 保存patch {patch_count + 1}: {patch_filename} (尺寸: 宽度{actual_width}×高度{actual_height})")
                patch_info.append({
                    'id': patch_count + 1,
                    'filename': patch_filename,
                    'coords': (x1, y1, x2, y2),
                    'size': (actual_width, actual_height)
                })
            else:
                print(f"❌ 保存失败: {patch_filename}")
                
            patch_count += 1
            
        if patch_count >= num_patches:
            break
    
    print(f"\n🎉 完成！共分割保存了 {patch_count} 个patch到文件夹: {output_folder}")
    
    # 返回patch信息，供可视化使用
    return image, patch_info, (rows, cols, patch_size)

def visualize_patches(image_path, output_folder, reshape_to_spectrogram=True):
    """
    可视化分割结果，在原图上绘制分割线
    """
    # 读取并处理图片（与split_image_into_patches保持一致）
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("❌ 无法读取图片进行可视化")
        return
    
    if reshape_to_spectrogram:
        image = reshape_image_to_spectrogram_size(original_image, 1024, 512)
    else:
        image = original_image
    
    height, width = image.shape[:2]  # height=1024, width=512
    
    # 使用与分割相同的布局计算
    if reshape_to_spectrogram and height == 1024 and width == 512:
        rows, cols = 4, 2
        patch_size = min(height // rows, width // cols)  # 256
    else:
        rows, cols = 4, 2
        patch_size = min(height // rows, width // cols)
    
    # 绘制分割线
    image_with_grid = image.copy()
    
    # 绘制垂直分割线
    for col in range(cols + 1):
        x = col * (width // cols)  # 0, 256, 512
        cv2.line(image_with_grid, (x, 0), (x, height), (0, 255, 0), 3)
    
    # 绘制水平分割线
    for row in range(rows + 1):
        y = row * (height // rows)  # 0, 256, 512, 768, 1024
        cv2.line(image_with_grid, (0, y), (width, y), (0, 255, 0), 3)
    
    # 绘制patch编号
    patch_count = 0
    for row in range(rows):
        for col in range(cols):
            if patch_count >= 8:
                break
            
            # 计算patch中心位置
            y1 = row * (height // rows)
            y2 = y1 + patch_size
            x1 = col * (width // cols)
            x2 = x1 + patch_size
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 绘制patch编号 - 白色背景，黑色文字
            text = str(patch_count + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            
            # 获取文字尺寸
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 绘制白色背景矩形
            cv2.rectangle(image_with_grid, 
                         (center_x - text_width//2 - 10, center_y - text_height//2 - 10),
                         (center_x + text_width//2 + 10, center_y + text_height//2 + 10),
                         (255, 255, 255), -1)
            
            # 绘制黑色文字
            cv2.putText(image_with_grid, text, 
                       (center_x - text_width//2, center_y + text_height//2), 
                       font, font_scale, (0, 0, 0), thickness)
            
            patch_count += 1
            
        if patch_count >= 8:
            break
    
    # 保存可视化结果
    grid_path = os.path.join(output_folder, "grid_visualization.png")
    cv2.imwrite(grid_path, image_with_grid)
    print(f"📊 可视化结果保存至: {grid_path}")

if __name__ == "__main__":
    # 设置路径
    image_path = r"D:\bishe\MVST-main\basevisualization\basecharts\1.png"
    output_folder = r"D:\bishe\MVST-main\basevisualization\basecharts\save"
    
    print("🚀 开始分割图片...")
    print(f"📂 输入图片: {image_path}")
    print(f"📁 输出文件夹: {output_folder}")
    print("-" * 50)
    
    # 执行分割
    split_image_into_patches(image_path, output_folder, num_patches=8, reshape_to_spectrogram=True)
    
    # 生成可视化
    print("\n📊 生成可视化...")
    visualize_patches(image_path, output_folder, reshape_to_spectrogram=True)
    
    print("\n✨ 所有任务完成！")