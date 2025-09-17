# test_gsplat_rendering.py
import gsplat

# 查看 rendering 子模块的所有接口
print("gsplat.rendering 子模块可用接口：")
for attr in dir(gsplat.rendering):
    if not attr.startswith("_"):
        print(f"- {attr}")

# 尝试查看核心渲染函数的签名（通常名为 render 或 render_gaussians）
try:
    from inspect import signature
    # 优先检查是否有 render 函数
    if hasattr(gsplat.rendering, "render"):
        print("\ngsplat.rendering.render 函数签名：")
        print(signature(gsplat.rendering.render))
    # 若没有，检查是否有 render_gaussians
    elif hasattr(gsplat.rendering, "render_gaussians"):
        print("\ngsplat.rendering.render_gaussians 函数签名：")
        print(signature(gsplat.rendering.render_gaussians))
except Exception as e:
    print(f"\n查看渲染函数签名失败：{e}")