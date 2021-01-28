#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
using namespace  std;
// 724. 寻找数组的中心索引 easy
// 2021年1月28日23点29分
//int main() {
//    vector<int> nums={1,7,3,6,5,6};
//    if(nums.empty())
//        return -1;
//    int all=nums[0];
//    vector<int> a;
//    a.push_back(0);
//    for(int i=1;i<nums.size();i++){
//        a.push_back(a.back()+nums[i-1]);
//        all+=nums[i];
//    }
//    for(int i=0;i<nums.size();i++){
//        if(all-nums[i]-a[i]==a[i]){
//            return i;
//        }
//    }
//    return -1;
//}
// 1631. 最小体力消耗路径 mid
// 2021年1月29日02点03分
// 最短路径spfa
//你准备参加一场远足活动。给你一个二维 rows x columns 的地图 heights ，其中 heights[row][col] 表示格子 (row, col) 的高度。一开始你在最左上角的格子 (0, 0) ，且你希望去最右下角的格子 (rows-1, columns-1) （注意下标从 0 开始编号）。你每次可以往 上，下，左，右 四个方向之一移动，你想要找到耗费 体力 最小的一条路径。
//一条路径耗费的 体力值 是路径上相邻格子之间 高度差绝对值 的 最大值 决定的。
//请你返回从左上角走到右下角的最小 体力消耗值 。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/path-with-minimum-effort

int main(){
    vector<vector<int> > heights={{1,10,6,7,9,10,4,9}};
    int n=heights.size();
    int m=heights[0].size();
    int dx[]{0,1,0,-1};
    int dy[]{1,0,-1,0};
    vector<vector<int> > ans(n,vector<int>(m,2147483647));
    ans[0][0]=0;
    queue<int> a;
    a.push(0);
    while (!a.empty()){
        int x = a.front()/m;
        int y = a.front()%m;
        a.pop();
        for(int i = 0;i<4;i++){
            int j = x+dx[i];
            int k = y+dy[i];
            if(j>=0&&j<n&&k>=0&&k<m) {
                int tmp = ans[x][y]>abs(heights[x][y] - heights[j][k])?ans[x][y]:abs(heights[x][y] - heights[j][k]);
                if (tmp < ans[j][k]) {
                    ans[j][k] = tmp;
                    a.push(j * m + k);
                }
            }
        }
    }
    return ans[n-1][m-1];
}
