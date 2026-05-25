为了提升allgather的性能，对原始代码在aicpu上进行了修改：
1. 新增了以下文件：
- src\ops\all_gather\template\aicpu\ins_temp_all_gather_mesh_clos_v2.cc
- src\ops\all_gather\template\aicpu\ins_temp_all_gather_mesh_clos_v2.h
