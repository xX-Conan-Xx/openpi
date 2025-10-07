from ikpy.chain import Chain

URDF_PATH = "/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"

# 读取 URDF 文件
chain = Chain.from_urdf_file(URDF_PATH)

# 遍历所有的 link 名称
print("🔍 URDF 中找到的 link 列表:")
for link in chain.links:
    print(f"- {link.name}")