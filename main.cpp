#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
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
//
//int main(){
//    vector<vector<int> > heights={{1,10,6,7,9,10,4,9}};
//    int n=heights.size();
//    int m=heights[0].size();
//    int dx[]{0,1,0,-1};
//    int dy[]{1,0,-1,0};
//    vector<vector<int> > ans(n,vector<int>(m,2147483647));
//    ans[0][0]=0;
//    queue<int> a;
//    a.push(0);
//    while (!a.empty()){
//        int x = a.front()/m;
//        int y = a.front()%m;
//        a.pop();
//        for(int i = 0;i<4;i++){
//            int j = x+dx[i];
//            int k = y+dy[i];
//            if(j>=0&&j<n&&k>=0&&k<m) {
//                int tmp = ans[x][y]>abs(heights[x][y] - heights[j][k])?ans[x][y]:abs(heights[x][y] - heights[j][k]);
//                if (tmp < ans[j][k]) {
//                    ans[j][k] = tmp;
//                    a.push(j * m + k);
//                }
//            }
//        }
//    }
//    return ans[n-1][m-1];
//}

//778. 水位上升的泳池中游泳 hard
//2021年1月20日22点10分
//最短路径spfa
//在一个 N x N 的坐标方格 grid 中，每一个方格的值 grid[i][j] 表示在位置 (i,j) 的平台高度。
//现在开始下雨了。当时间为 t 时，此时雨水导致水池中任意位置的水位为 t 。你可以从一个平台游向四周相邻的任意一个平台，但是前提是此时水位必须同时淹没这两个平台。假定你可以瞬间移动无限距离，也就是默认在方格内部游动是不耗时的。
//当然，在你游泳的时候你必须待在坐标方格里面。
//你从坐标方格的左上平台 (0，0) 出发。最少耗时多久你才能到达坐标方格的右下平台 (N-1, N-1)？
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/swim-in-rising-water
//int main() {
//    vector<vector<int> > grid={{1,10,6,7,9,10,4,9}};
//    int n = grid.size();
//    int m = grid[0].size();
//    int dx[]{0, 1, 0, -1};
//    int dy[]{1, 0, -1, 0};
//    vector<vector<int> > ans(n, vector<int>(m, 2147483647));
//    ans[0][0] = grid[0][0];
//    queue<int> a;
//    a.push(0);
//    while (!a.empty()) {
//        int x = a.front() / m;
//        int y = a.front() % m;
//        a.pop();
//        for (int i = 0; i < 4; i++) {
//            int j = x + dx[i];
//            int k = y + dy[i];
//            if (j >= 0 && j < n && k >= 0 && k < m) {
//                int tmp = ans[x][y] > abs(grid[j][k]) ? ans[x][y] : abs(grid[j][k]);
//                if (tmp < ans[j][k]) {
//                    ans[j][k] = tmp;
//                    a.push(j * m + k);
//                }
//            }
//        }
//    }
//    return ans[n - 1][m - 1];
//}

//839. 相似字符串组 hard
//2021年1月31日20点24分
//并查集
//如果交换字符串 X 中的两个不同位置的字母，使得它和字符串 Y 相等，那么称 X 和 Y 两个字符串相似。如果这两个字符串本身是相等的，那它们也是相似的。
//例如，"tars" 和 "rats" 是相似的 (交换 0 与 2 的位置)； "rats" 和 "arts" 也是相似的，但是 "star" 不与 "tars"，"rats"，或 "arts" 相似。
//总之，它们通过相似性形成了两个关联组：{"tars", "rats", "arts"} 和 {"star"}。注意，"tars" 和 "arts" 是在同一组中，即使它们并不相似。形式上，对每个组而言，要确定一个单词在组中，只需要这个词和该组中至少一个单词相似。
//给你一个字符串列表 strs。列表中的每个字符串都是 strs 中其它所有字符串的一个字母异位词。请问 strs 中有多少个相似字符串组？
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/similar-string-groups

//vector<int> isbl;
//
//int ab(string a,string b){
//    int num = 0;
//    for (int i = 0; i <a.size(); i++) {
//        if (a[i] != b[i]) {
//            num++;
//            if (num > 2) {
//                return false;
//            }
//        }
//    }
//    return true;
//}
//
//int find(int x){
//    return isbl[x] == x ? x : find(isbl[x]);
//}
//
//int main(){
//    vector<string> strs = {"ajdidocuyh","djdyaohuic","ddjyhuicoa","djdhaoyuic","ddjoiuycha","ddhoiuycja","ajdydocuih","ddjiouycha","ajdydohuic","ddjyouicha"};
//    isbl.resize(strs.size());
//    for (int i = 0; i < strs.size(); i++) {
//        isbl[i] = i;
//    }
//    int ans=0;
//    for(int j=0;j<strs.size();j++){
//        for(int k=j+1;k<strs.size();k++){
//            int jl=find(j);
//            int kl=find(k);
//            if(jl==kl) continue;
//            if(ab(strs[j],strs[k])) isbl[kl]=isbl[j]; //这里若改成k，则整个kl块会被忽略掉更新，其他元素指向祖先，所以应该祖先合并才能逐步跳合并
//        }
//    }
//    int ret = 0;
//    for (int i = 0; i < strs.size(); i++) {
//        if (isbl[i] == i) {
//            ans++;
//        }
//    }
//    return ans;
//}

//888. 公平的糖果棒交换 easy
//2021年2月1日0点34分
//爱丽丝和鲍勃有不同大小的糖果棒：A[i] 是爱丽丝拥有的第 i 根糖果棒的大小，B[j] 是鲍勃拥有的第 j 根糖果棒的大小。
//因为他们是朋友，所以他们想交换一根糖果棒，这样交换后，他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有的糖果棒大小的总和。）
//返回一个整数数组 ans，其中 ans[0] 是爱丽丝必须交换的糖果棒的大小，ans[1] 是 Bob 必须交换的糖果棒的大小。
//如果有多个答案，你可以返回其中任何一个。保证答案存在。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/fair-candy-swap
//int main(){
//    vector<int> A{1,1};
//    vector<int> B{2,2};
//    map<int,int> c;
//    map<int,int> d;
//    int sumA=0;
//    int sumB=0;
//    for(auto i :A){
//        sumA+=i;
//        c[i]=1;
//    }
//    for(auto i:B){
//        sumB+=i;
//        d[i]=1;
//    }
//    int cz = sumA-sumB;
//    vector<int> ans;
//    for(int i:A){
//        if(d.find((cz-2*i)/-2)!=d.end()) {
//            ans.push_back(i);
//            ans.push_back((cz-2*i)/-2);
//            break;
//        }
//    }
//    cout<<ans[0]<<ans[1];
//}

//424. 替换后的最长重复字符 mid
//2021年2月2日21点10分
//给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。
//注意：字符串长度 和 k 不会超过 10^4。
//示例 1：
//输入：s = "ABAB", k = 2
//输出：4
//解释：用两个'A'替换为两个'B',反之亦然。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/longest-repeating-character-
int main(){
    string s = "AABABBA";
    int k =1;
    vector<int> num(26);
    int maxn=0;
    int left=0,right=0;
    while (right<s.length()){
        num[s[right]-'A']++;
        maxn= max(maxn,num[s[right]-'A']);
        if(right-left+1-maxn<k){
            num[s[left]-'A']--;
            left++;
        }
        right++;
    }
    return right-left;
}