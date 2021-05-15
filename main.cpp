#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <stack>
#include <cstdio>
#include <algorithm>
#include <bitset>
#include <cstring>
#include <set>
#include <string>
#include <sstream>
#include <limits>
#include "SortTestHelper.h"
#include "Heap.h"


#define INF 0x3f3f3f3f
#define pb push_back
using LL = long long;
using pii = pair<int, int>;

inline int lowbit(int x) { return x & (-x); }

template<typename A>
inline A __lcm(A a, A b) { return a / __gcd(a, b) * b; }

template<typename A, typename B, typename C>
inline A fpow(A x, B p, C lyd) {
    A ans = 1;
    for (; p; p >>= 1, x = 1LL * x * x % lyd)if (p & 1)ans = 1LL * x * ans % lyd;
    return ans;
}

using namespace std;

int gcd(int a, int b) {
    return b > 0 ? gcd(b, a % b) : a;
}

int lcm(int a, int b) {
    return a * b / gcd(a, b);
}
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
//int main(){
//    string s = "AABABBA";
//    int k =1;
//    vector<int> num(26);
//    int maxn=0;
//    int left=0,right=0;
//    while (right<s.length()){
//        num[s[right]-'A']++;
//        maxn= max(maxn,num[s[right]-'A']);
//        if(right-left+1-maxn<k){
//            num[s[left]-'A']--;
//            left++;
//        }
//        right++;
//    }
//    return right-left;
//}

//随便写写

//int main(){
//    int i;
//    cin>>i;
//    int a[i];
//    for(int z=0;z<i;z++){
//        cin>>a[z];
//    }
//    int ans=0;
//    ans=gcd(a[0],a[1]);
//    for(int z=2;z<i;z++){
//        ans=gcd(ans,a[z]);
//    }
//    cout<<ans;
//}
//int main(){
//    stack<int> a;
//    int n;
//    scanf("%d",&n);
//    for(int i=0;i<n;i++){
//        int b,c;
//        scanf("%d",&b);
//        if(b==1){
//            scanf("%d",&c);
//            a.push(c);
//        }
//        if(b==2){
//            scanf("%d",&c);
//            for(int j=0;j<c;j++)
//                a.pop();
//        }
//        if(b==3){
//            printf("%d\n",a.top());
//        }
//    }
//}

//480. 滑动窗口中位数 hard
//2021年2月3日10点55分
//中位数是有序序列最中间的那个数。如果序列的大小是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。
//例如：
//[2,3,4]，中位数是 3
//[2,3]，中位数是 (2 + 3) / 2 = 2.5
//给你一个数组 nums，有一个大小为 k 的窗口从最左端滑动到最右端。窗口中有 k 个数，每次窗口向右移动 1 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/sliding-window-median
//python偷懒，暴力能过，等下再写个不暴力的
//class Solution:
//        def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
//return [sorted(nums[i:i+k])[k//2] for i in range(len(nums)-k+1)] if k%2 else [(sorted(nums[i:i+k])[k//2]+sorted(nums[i:i+k])[(k-1)//2])/2 for i in range(len(nums)-k+1)]

//面试题 01.01. 判定字符是否唯一
//2021年2月3日20点55分
//实现一个算法，确定一个字符串 s 的所有字符是否全都不同。
//示例 1：
//输入: s = "leetcode"
//输出: false
//示例 2：
//输入: s = "abc"
//输出: true
//限制：
//0 <= len(s) <= 100
//如果你不使用额外的数据结构，会很加分。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/is-unique-lcci
//int main(){
//    string astr;
//    int a[260]={0};
//    for(char i:astr){
//        if(a[i]>0)
//            return false;
//        a[i]++;
//    }
//    return true;
//}

//剑指 Offer 09. 用两个栈实现队列 easy
//2021年2月3日23点35分
//用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof
//class CQueue {
//public:
//    stack<int > a;
//    stack<int > b;
//    CQueue() {
//    }
//
//    void appendTail(int value) {
//        a.push(value);
//    }
//
//    int deleteHead() {
//        if(!b.empty()){
//            int ans = b.top();
//            b.pop();
//            return ans;
//        }else{
//            while(!a.empty()){
//                b.push(a.top());
//                a.pop();
//            }
//        }
//        if(!b.empty()){
//            int ans = b.top();
//            b.pop();
//            return ans;
//        }
//        return -1;
//    }
//};

//643. 子数组最大平均数 I easy
//2021年2月4日00点10分
//给定 n 个整数，找出平均数最大且长度为 k 的连续子数组，并输出该最大平均数。
//示例：
//输入：[1,12,-5,-6,50,3], k = 4
//输出：12.75
//解释：最大平均数 (12-5-6+50)/4 = 51/4 = 12.75
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/maximum-average-subarray-i
//int main(){
//    vector<int> nums;
//    int k;
//    int ans=0;
//    int s=0;
//    for(int i=0;i<k;i++){
//        s+=nums[i];
//    }
//    ans=s;
//    for(int i=k;i<nums.size();i++){
//        s+=nums[i];
//        s-=nums[i-k];
//        ans=max(ans,s);
//    }
//    return double(ans)/double(k);
//}


//剑指 Offer 10- II. 青蛙跳台阶问题 easy
//2021年2月4日09点50分
//一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
//答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof
//int main(){
//    int n=4;
//    if(n==0||n==1)
//        return 1;
//    int a=1;
//    int b=1;
//    int c=0;
//    for(int i=1;i<n;i++){
//        a=(a+b)%1000000007;
//        swap(a,b);
//    }
//    return b;
//}


//1208. 尽可能使字符串相等 mid
//2021年2月5日1点34分
//给你两个长度相同的字符串，s 和 t。
//将 s 中的第 i 个字符变到 t 中的第 i 个字符需要 |s[i] - t[i]| 的开销（开销可能为 0），也就是两个字符的 ASCII 码值的差的绝对值。
//用于变更字符串的最大预算是 maxCost。在转化字符串时，总开销应当小于等于该预算，这也意味着字符串的转化可能是不完全的。
//如果你可以将 s 的子字符串转化为它在 t 中对应的子字符串，则返回可以转化的最大长度。
//如果 s 中没有子字符串可以转化成 t 中对应的子字符串，则返回 0。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/get-equal-substrings-within-budget
//int main(){
//    string s="abcd";
//    string t="cdef";
//    int maxCost=3;
//    int left=0;
//    int right=0;
//    int ans=0;
//    int tmp=0;
//    while (right<s.size()){
//        if(tmp>maxCost){
//            tmp-=abs(s[left]-t[left]);
//            ans=max(ans,right-left-1);
//            left++;
//        }
//        else{
//            tmp+=abs(s[right]-t[right]);
//            right++;
//        };
//    }
//    ans = max(ans, tmp <= maxCost ? right - left : right - left - 1);
//    return ans;
//}


//1423. 可获得的最大点数 mid
//2021年2月6日20点05分
//几张卡牌 排成一行，每张卡牌都有一个对应的点数。点数由整数数组 cardPoints 给出。
//每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 k 张卡牌。
//你的点数就是你拿到手中的所有卡牌的点数之和。
//给你一个整数数组 cardPoints 和整数 k，请你返回可以获得的最大点数。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards
//int main(){
//    vector<int> cardPoints = {1,2,3,4,5,6,1};
//    int k=3;
//    int ans=0;
//    int tmp;
//    for(int i=cardPoints.size()-k;i<cardPoints.size();i++){
//        ans+=cardPoints[i];
//    }
//    tmp=ans;
//    for(int i=0;i<k;i++){
//        tmp+=(cardPoints[i]-cardPoints[cardPoints.size()-k+i]);
//        ans=max(tmp,ans);
//    }
//    return ans;
//}


//665. 非递减数列
//int main(){
//    vector<int> nums;
//    int flag=0;
//    for(int i=1;i<nums.size()&&flag<2;i++){
//        if(nums[i-1]>=nums[i]){
//            flag++;
//            if(i-2>=0&&nums[i-2]>nums[i])
//                nums[i]=nums[i-1];
//            else
//                nums[i-1]=nums[i];
//        }
//    }
//    return flag<=1;
//}


//5657. 唯一元素的和
//int main(){
//    int a[106]={0};
//    for(int i=0;i<nums.size();i++)
//        a[nums[i]]++;
//    int ans=0;
//    for(int i=0;i<103;i++){
//        if(a[i]==1)
//            ans+=i;
//    }
//    return ans;
//}


//5658. 任意子数组和的绝对值的最大值
//s = [0]
//for i in nums:
//s.append(s[-1]+i)
//return abs(min(s)-max(s))

//忘了啥题
//int main(){
//    string s;
//    int n =s.length();
//    int j=0;
//    int k=n-1;
//    while (j<k&&s[j]==s[k]){
//        char c=s[j];
//        while (c==s[j]&&j<=k) j++;
//        while (c==s[k]&&j<=k) k--;
//    }
//    return k-j-1;
//}


//5672. 检查数组是否经排序和轮转得到
//int main(){
//    vector<int> nums = {6,10,6};
//    vector<int> tmp(nums);
//    sort(tmp.begin(),tmp.end());
//    vector<int> min;
//    for(int i=0;i<nums.size();i++){
//        if(nums[i]==tmp[0]){
//            min.push_back(i);
//        }
//
//    }
//    int tmp1=0;
//    int flag=1;
//    for(int j=0;j<min.size();j++){
//        for(int i=0;i<nums.size();i++){
//            tmp1=(i+min[j])%nums.size();
//            if(tmp[i]!=nums[tmp1]){
//                flag=0;
//                break;
//            }
//        }
//        if(flag)
//            return true;
//        else{
//            flag=1;
//        }
//    }
//    return false;
//}


//5673. 移除石子的最大得分
//int main(){
//    int a=4,b=4,c=6;
//    priority_queue <int,vector<int>,less<int> > q;
//    q.push(a);
//    q.push(b);
//    q.push(c);
//    int ans=0;
//    while(q.size()>=2){
//        int tmp1=q.top()-1;
//        q.pop();
//        int tmp2=q.top()-1;
//        q.pop();
//        if(tmp1!=0)
//            q.push(tmp1);
//        if(tmp2!=0)
//            q.push(tmp2);
//        ans++;
//    }
//    return ans;
//}


//5674. 构造字典序最大的合并字符串
//int main(){
//    string word1 = "cabaa";
//    string word2 = "bcaaa";
//    int j=0;
//    int k=0;
//    string ans;
//    while (j<word1.size()||k<word2.size()){
//        if(j<word1.size()&&k<word2.size()){
//            char tmp;
//            if(word1[j]>word2[k]){
//                tmp=word1[j];
//                j++;
//            } else if(word1[j]==word2[k]){
//                int tmp1=j,tmp2=k;
//                int flag=0;
//                while(tmp1<word1.size()&&tmp2<word2.size()){
//                    if(word1[tmp1]!=word2[tmp2])
//                        break;
//                    tmp1++;
//                    tmp2++;
//                }
//                if(word1[tmp1]>word2[tmp2])
//                {
//                    tmp=word1[j];
//                    j++;
//                } else if(word1[tmp1]<word2[tmp2]){
//                    tmp=word2[k];
//                    k++;
//                }else{
//                    if(tmp1<word1.size()){
//                        tmp=word1[j];
//                        j++;
//                    } else{
//                        tmp=word2[k];
//                        k++;
//                    }
//                }
//            }else{
//                tmp=word2[k];
//                k++;
//            }
//            ans.push_back(tmp);
//        }
//        else if(j<word1.size()){
//            for(int i=j;i<word1.size();i++){
//                ans.push_back(word1[i]);
//            }
//            break;
//        }else{
//            for(int i=k;i<word2.size();i++){
//                ans.push_back(word2[i]);
//            }
//            break;
//        }
//    }
//    cout<<ans;
//}

//int main(){
//    vector<int> nums = {7,-9,15,-2};
//    int goal=-5;
//    vector<int> tmp={0};
//    for(int i=0;i<=nums.size();i++){
//        tmp.push_back(tmp[i]+nums[i]);
//    }
//    int ans=999999999;
//    for(int j=0;j<nums.size();j++){
//        for(int k=j+1;k<nums.size();k++){
//            ans=min(ans,abs(tmp[k]-tmp[j]-goal));
//        }
//    }
//    return ans;
//}
//978. 最长湍流子数组
//int len = A.size();
//        int up = 1, down = 1;
//        int ans = 1;
//        for (int i = 1; i < len; i++) {
//            if (A[i] > A[i - 1]) { up = down + 1; down = 1; }
//            else if (A[i] < A[i - 1]) { down = up + 1; up = 1; }
//            else { up = down = 1; }
//            ans = max(ans, max(up, down));
//        }
//        return ans;
//992. K 个不同整数的子数组
//class Solution {
//public int subarraysWithKDistinct(int[] A, int K) {
//        return subarrayNomorethanK(A,K)-subarrayNomorethanK(A,K-1);
//    }
//private int subarrayNomorethanK(int[] A, int k) {//不同元素不超过k的子数组个数
//        int l=0,r=0;
//        int ans=0;
//        int[] count=new int[A.length+1];//记录每个整数出现的频数。A[i]取值在[1,A.length]
//        int diffElem=0;//count[]非零元素个数，即不同的整数个数
//        while(r<A.length){
//            if(count[A[r]]==0) diffElem++;
//            count[A[r++]]++;//右滑窗口
//            while(diffElem>k){//窗口内的不同整数个数超过了K，缩小窗口
//                count[A[l]]--;
//                if(count[A[l]]==0)diffElem--;
//                l++;
//            }
//            ans+=r-l;//窗口内所有的“不同整数不超过k的子数组个数”(左闭右开)
//        }
//        return ans;
//    }
//}

//561. 数组拆分 I
//给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大。
//返回该 最大总和 。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/array-partition-i
//int main(){
//    vector<int> nums={1,4,3,2};
//    sort(nums.begin(),nums.end());
//    int ans=0;
//    for(int i=0;i<nums.size();i+=2)
//        ans+=nums[i];
//    return ans;
// }

//485. 最大连续 1 的个数
//int main(){
//    vector<int> nums = {1,0,1,1,0,1};
//    int ans = 0;
//    int tmp = 0;
//    for(int i=0;i<nums.size();i++)
//        if(nums[i])
//            tmp++;
//        else{
//            ans=max(tmp,ans);
//            tmp=0;
//        }
//    return max(tmp,ans);;
//}

//448. 找到所有数组中消失的数字
//给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
//找到所有在 [1, n] 范围之间没有出现在数组中的数字。
//您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array
//int main(){
//    vector<int> nums= {4,3,2,7,8,2,3,1};
//    for(int i=0;i<nums.size();i++){
//        if(nums[abs(nums[i])-1]>0)
//            nums[abs(nums[i])-1]*=-1;
//    }
//    vector<int > ans;
//    for(int i=0;i<nums.size();i++){
//        if(nums[i]>0)
//            ans.push_back(i+1);
//    }
//    for(int i=0;i<ans.size();i++){
//        cout<<ans[i]<<" ";
//    }
//}


//119. 杨辉三角 II
//int main(){
//    int rowIndex;
//    vector<int> res(rowIndex+1);
//    res[0] = 1;
//    for(int j = 1; j <= rowIndex; j++){
//        for(int i = j; i >=0; i--){
//            res[i] = (i-1>=0?res[i-1]: 0) + res[i];
//        }
//    }
//
//    return res;
//}

//1004. 最大连续1的个数 III
//给定一个由若干 0 和 1 组成的数组 A，我们最多可以将 K 个值从 0 变成 1 。
//返回仅包含 1 的最长（连续）子数组的长度
//int main(){
//    vector<int> A={0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1};
//    int K=3;
//    int left =0 , right=0 , len=A.size(),cut=0,ans=0;
//    while (right<len){
//        if(A[right]==1)
//            right++;
//        else{
//            if(cut<K){
//                cut++;
//                right++;
//            } else{
//                if(A[left]==0)
//                    cut--;
//                left++;
//            }
//        }
//        ans = max(ans,right-left);
//    }
//    return ans;
//}


//697. 数组的度
//给定一个非空且只包含非负数的整数数组 nums，数组的度的定义是指数组里任一元素出现频数的最大值。
//你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/degree-of-an-array
//int main(){
//    vector<int> nums = {1,2,2,3,1,4,2};
//    map<int,vector<int>> a;
//    for(int i=0;i<nums.size();i++){
//
//        if(a.find(nums[i])==a.end()){
//            vector<int> z(3,0);
//            a.insert(make_pair(nums[i],z));
//            auto& p = a[nums[i]];
//            p[0]++;
//            p[1]=i;
//        }else{
//            auto& p = a[nums[i]];
//            p[0]++;
//            p[2]=i;
//        }
//    }
//    int ans=500010,tmp=2;
//    for(auto i=a.begin();i!=a.end();i++){
//        if(i->second[0]>=tmp){
//            if(i->second[0]>=tmp){
//                if(i->second[0]==tmp)
//                    ans=min(ans,i->second[2]-i->second[1]);
//                else{
//                    tmp = i->second[0];
//                    ans=i->second[2]-i->second[1];
//                }
//            }
//        }
//    }
//    if(ans==500010)
//        return 1;
//    return ans+1;
//}


//1438. 绝对差不超过限制的最长连续子数组
//给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 limit 。
//如果不存在满足条件的子数组，则返回 0 。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit
//class Solution {
//public int longestSubarray(int[] nums, int limit) {deque<int> min_que, max_que;
//        int left = 0, right = 0, ans = 0;
//        while(right < nums.size())
//        {
//            // 滑动窗口的最小值，单调递增
//            while(!min_que.empty() && min_que.back() > nums[right])
//                min_que.pop_back();
//            min_que.push_back(nums[right]);
//            // 滑动窗口的最大值，单调递减
//            while(!max_que.empty() && max_que.back() < nums[right])
//                max_que.pop_back();
//            max_que.push_back(nums[right]);
//            right++;
//            // 根据窗口的最小值和最大值的差更新窗口
//            while(max_que.front()-min_que.front() > limit)
//            {
//                if(nums[left] == min_que.front())
//                    min_que.pop_front();
//                if(nums[left] == max_que.front())
//                    max_que.pop_front();
//                left++;
//            }
//            ans = max(ans, right-left);
//        }
//        return ans
//    }
//}


//766. 托普利茨矩阵
//给你一个 m x n 的矩阵 matrix 。如果这个矩阵是托普利茨矩阵，返回 true ；否则，返回 false 。
//如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是 托普利茨矩阵 。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/toeplitz-matrix
//int main(){
//    vector<vector<int>> matrix={{1,2,3,4},{5,1,2,3},{9,5,1,2}};
//    for(int i=0;i<matrix.size()-1;i++){
//        for(int j=0;j<matrix[0].size()-1;j++){
//            if(matrix[i][j]!=matrix[i+1][j+1]){
//                return false;
//            }
//        }
//    }
//    return true;
//}


//1052. 爱生气的书店老板
//今天，书店老板有一家店打算试营业 customers.length 分钟。每分钟都有一些顾客（customers[i]）会进入书店，所有这些顾客都会在那一分钟结束后离开。
//在某些时候，书店老板会生气。 如果书店老板在第 i 分钟生气，那么 grumpy[i] = 1，否则 grumpy[i] = 0。 当书店老板生气时，那一分钟的顾客就会不满意，不生气则他们是满意的。
//书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续 X 分钟不生气，但却只能使用一次。
//请你返回这一天营业下来，最多有多少客户能够感到满意的数量
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/grumpy-bookstore-owner
//int main(){
//    vector<int> customers={3,8,8,7,1};
//    vector<int> grumpy = {1,1,1,1,1};
//    int X=3;
//    vector<int> a(customers.size());
//    vector<int> b(customers.size());
//    a[0]=customers[0];
//    b[0]=(!grumpy[0])?customers[0]:0;
//    for(int i=1;i<customers.size();i++){
//        a[i]=a[i-1]+customers[i];
//        b[i]=b[i-1]+(!grumpy[i]?customers[i]:0);
//    }
//    if(X+1>customers.size()-1){
//        if(X>=customers.size()-1)
//            return a[customers.size()-1];
//        return b[customers.size()-1];
//    }
//    int ans=a[X-1]+b[customers.size()-1]-b[X-1];
//    for(int i=0;i<customers.size()-X;i++){
//        ans=max(ans,b[i]+a[i+X]-a[i]+b[customers.size()-1]-b[i+X]);
//    }
//    ans=max(ans,b[customers.size()-1-X]+a[customers.size()-1]-a[customers.size()-1-X]);
//    return ans;
//}


//牛客网
//struct a{
//    int x;
//    int y;
//};
//
//bool cmd(a a1,a a2){
//    if(a1.x*a1.y!=a2.x*a2.y)
//        return a1.x*a1.y<a2.x*a2.y;
//    else{
//        if(a1.x==a1.y){
//            return 1;
//        }else if(a2.x==a2.y){
//            return 0;
//        } else if(min(double (a1.x)/double(a1.y),double (a1.y)/double (a1.x))!=min(double (a2.x)/double (a2.y),double (a2.y)/double (a2.x)))
//            return min(double (a1.x)/double(a1.y),double (a1.y)/double (a1.x))>min(double (a2.x)/double (a2.y),double (a2.y)/double (a2.x));
//        else{
//            return a1.x>a2.x;
//        }
//    }
//}
//
//int main(){
//    int n;
//    cin>>n;
//    a a[n];
//    for(int i=0;i<n;i++){
//        cin>>a[i].x>>a[i].y;
//    }
//    sort(a,a+n,cmd);
//    cout<<a[0].x<<" "<<a[0].y;
//    for(int i=1;i<n;i++){
//        cout<<' '<<a[i].x<<" "<<a[i].y;
//    }
//}


//832. 翻转图像
//给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。
//水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。
//反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/flipping-an-image
//int main(){
//    return [(1-i for i in row[::-1]) for row in A]
//    //pykuaile
//}


//867. 转置矩阵
//给你一个二维整数数组 matrix， 返回 matrix 的 转置矩阵 。
//矩阵的 转置 是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。
//int main(){
//    return [list(x) for x in zip(*matrix)]
//    //py快乐日
//}


//53. 最大子序和
// 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
//int main(){
//    vector<int> nums={-2,1,-3,4,-1,2,1,-5,4};
//    int ans=0;
//    int tmp=0;
//    int m = -1000000;
//    for(int i=0;i<nums.size();i++){
//        if(tmp+nums[i]<0)
//            tmp =0;
//        else
//        tmp = tmp+nums[i];
//      ans=max(ans,tmp);
//      m=max(m,nums[i]);
//    }
//    if(m<0)
//        return m;
//    return ans;
//}


//461. 汉明距离
//两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
//给出两个整数 x 和 y，计算它们之间的汉明距离。
//注意：
//0 ≤ x, y < 231.
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/hamming-distance
//int main(){
//    int x,y;
//    x = x^y;
//    y=0;
//    while(x>0){
//        y += x%2;
//        x/=2;
//    }
//    return y;
//}

//896. 单调数列
//如果数组是单调递增或单调递减的，那么它是单调的。
//如果对于所有 i <= j，A[i] <= A[j]，那么数组 A 是单调递增的。 如果对于所有 i <= j，A[i]> = A[j]，那么数组 A 是单调递减的。
//当给定的数组 A 是单调数组时返回 true，否则返回 false。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/monotonic-array
//int main(){
//    vector<int> A;
//    int a=0,b=0;
//    if(A.size()==1)
//        return true;
//    for(int i=1;i<A.size();i++){
//        if(A[i-1]<A[i])a=1;
//        if(A[i-1]>A[i])b=1;
//    }
//    if(a+b==2)return false;
//    return true;
//}

//5689. 统计匹配检索规则的物品数量
//给你一个数组 items ，其中 items[i] = [typei, colori, namei] ，描述第 i 件物品的类型、颜色以及名称。
//另给你一条由两个字符串 ruleKey 和 ruleValue 表示的检索规则。
//如果第 i 件物品能满足下述条件之一，则认为该物品与给定的检索规则 匹配 ：
//ruleKey == "type" 且 ruleValue == typei 。
//ruleKey == "color" 且 ruleValue == colori 。
//ruleKey == "name" 且 ruleValue == namei 。
//统计并返回 匹配检索规则的物品数量 。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/count-items-matching-a-rule
//int main(){
//        vector< vector<string> > items;
//        string ruleKey;
//        string ruleValue;
//        int c=0;
//        if(ruleKey=="type"){
//            c=0;
//        }else if(ruleKey=="color"){
//            c=1;
//        }else{
//            c=2;
//        }
//        int ans;
//        for(int i=0;i<items.size();i++){
//            if(items[i][c]==ruleValue)
//                ans++;
//        }
//    return ans;
//};


//int main(){
//    vector<int> nums1;
//    vector<int> nums2;
//    int tmp11[6]={0};
//    int tmp22[6]={0};
//    int tmp1=0;
//    int tmp2=0;
//    for(int i=0;i<nums1.size();i++){
//        tmp1+=nums1[i];
//        tmp11[nums1[i]]++;
//    }
//    for(int i=0;i<nums2.size();i++){
//        tmp2+=nums2[i];
//        tmp22[nums2[i]]++;
//    }
//    int tmp = tmp1-tmp2;
//}

//5690. 最接近目标价格的甜点成本
//你打算做甜点，现在需要购买配料。目前共有 n 种冰激凌基料和 m 种配料可供选购。而制作甜点需要遵循以下几条规则：
//必须选择 一种 冰激凌基料。
//可以添加 一种或多种 配料，也可以不添加任何配料。
//每种类型的配料 最多两份 。
//给你以下三个输入：
//baseCosts ，一个长度为 n 的整数数组，其中每个 baseCosts[i] 表示第 i 种冰激凌基料的价格。
//toppingCosts，一个长度为 m 的整数数组，其中每个 toppingCosts[i] 表示 一份 第 i 种冰激凌配料的价格。
//target ，一个整数，表示你制作甜点的目标价格。
//你希望自己做的甜点总成本尽可能接近目标价格 target 。
//返回最接近 target 的甜点成本。如果有多种方案，返回 成本相对较低 的一种。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/closest-dessert-cost
//int main(){
//    vector<int> baseCosts;
//    vector<int> toppingCosts;
//    int target;
//    bitset<20005> can;
//    for(auto x:baseCosts) can[x]=1;
//    for(auto x:toppingCosts) can|=can<<x|(can<<(2*x));
//    int ans=-200000;
//    for(int x=0;x<20000;x++){
//     if(can[x]&&abs(x-target)<abs(ans-target))
//         ans=x;
//    }
//    return ans;
//}

//5691. 通过最少操作次数使数组的和相等
//给你两个长度可能不等的整数数组 nums1 和 nums2 。两个数组中的所有值都在 1 到 6 之间（包含 1 和 6）。
//每次操作中，你可以选择 任意 数组中的任意一个整数，将它变成 1 到 6 之间 任意 的值（包含 1 和 6）。
//请你返回使 nums1 中所有数的和与 nums2 中所有数的和相等的最少操作次数。如果无法使两个数组的和相等，请返回 -1 。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations
//int main(){
//    vector<int> a=;
//    vector<int> b;
//    sort(a.begin(),a.end());
//    sort(b.begin(),b.end());
//    int ans=0;
//    int aa=accumulate(a.begin(),a.end(),0);
//    int bb=accumulate(b.begin(),b.end(),0);
//    if (aa>bb)
//        swap(aa,bb),swap(a,b);
//    int j=0,k=b.size()-1;
//    while(j<a.size()&&k>=0&&aa<bb){
//        if(6-a[j]>b[k]-1){
//            aa+=6-a[j++];
//        }else{
//            bb-=b[k--]-1;
//        }
//        ans++;
//    }
//    while(j<a.size()&&aa<bb){
//        aa+=6-a[j++];
//        ans++;
//    }
//    while(k>=0&&aa<bb){
//        bb-=b[k--]-1;
//        ans++;
//    }
//    return aa>=bb?ans:-1;
//}


//int main(){
//    int tmp[9][9];
//    for(int i=0;i<81;i++){
//        cin>>tmp[i/9][i%9];
//    }
//    int ans=0;
//    for(int i=0;i<9;i++){
//        vector<int > a(9,0);
//        for(int j=0;j<9;j++){
//            a[tmp[i][j]-1]++;
//        }
//        for(int q=0;q<9;q++){
//            if(a[q]==0){
//                ans++;
//                break;
//            }
//        }
//    }
//    for(int i=0;i<9;i++) {
//        vector<int > a(9,0);
//        for (int j = 0; j < 9; j++) {
//            a[tmp[j][i] - 1]++;
//        }
//        for (int q = 0; q < 9; q++) {
//            if (a[q] == 0){
//                ans++;
//                break;
//            }
//        }
//    }
//    vector<int> tmp2={0,3,6,27,30,33,54,57,60};
//    for(int i:tmp2){
//        vector<int > a(9,0);
//        for(int j=0;j<3;j++){
//            for(int k=0;k<3;k++){
//                a[tmp[(i+j*9+k)/9][(i+j*9+k)%9]-1]++;
//            }
//            for (int q = 0; q < 9; q++) {
//                if (a[q] == 0){
//                    ans++;
//                    break;
//                }
//
//            }
//        }
//    }
//    cout<<ans;
//}

//int main()
//{
//	double eps = 1e-6;
//	double k;
//	cin>>k;
//	double l = 0.0,r,mid;
//	if (k>=1) r = k;
//    if (k<1)  r = 1;
//	while (fabs(l-k/l)>eps)
//	{
//		mid = l + (r - l) /2 ;
//		if (mid<k/mid)
//		{
//			l = mid;
//		}
//		else {
//			r = mid;
//		}
//	}
//	printf("%.3f",l);
//	return 0;
//}

//int searchInsert(int* nums, int numsSize, int target)
//{
//    int dst = 0;
//    while (dst<numsSize)
//    {
//        if (nums[dst] < target)
//        {
//            ++dst;
//        }
//        else
//        {
//            break;
//        }
//    }
//    return dst;
//}
//int main(){
//    int  a[INT16_MAX-1];
//    int z=0;
//    int number;
//    while (1) {
//        cin >> number;
//        a[z]=number;//每输入一个数字就把它添加到数组的最后
//        if (cin.get() == '\n')//如果是回车符则跳出循环
//            break;
//    }
//
//    int t;
//    cin>>t;
//    cout<<searchInsert(a,z,t);
//}



//int main()
//{
//    string str, res;
//    cin >> str;
//    int index;
//    if(str.size()>2){
//        for (int i = 1; i < str.size(); i++) {
//            if (str.substr(0, i) == str.substr(str.size() - i, i))
//                index = i;
//        }
//        cout << str + str.substr(index, str.size() - index);
//    }
//    else{
//        if(str.size()==1){
//            cout<<str;
//        }else{
//            if(str[0]==str[1]){
//                cout<<str;
//            }else{
//                cout<<str<<str;
//            }
//        }
//    }
//
//}


//304. 二维区域和检索 - 矩阵不可变
//给定一个二维矩阵，计算其子矩形范围内元素的总和，该子矩阵的左上角为 (row1, col1) ，右下角为 (row2, col2) 。
//上图子矩阵左上角 (row1, col1) = (2, 1) ，右下角(row2, col2) = (4, 3)，该子矩形内元素的总和为 8。
//来源：力扣（LeetCode）
//链接：https://leetcode-cn.com/problems/range-sum-query-2d-immutable
//class NumMatrix {
//public:
//    vector<vector<int>> sum;
//    NumMatrix(vector<vector<int>>& matrix) {
//        if(matrix.size()==0 || matrix[0].size()==0){
//            return ;
//        }
//        sum = vector<vector<int>>(matrix);
//        for(int i=1;i<matrix.size();i++){
//            sum[i][0] += sum[i-1][0];
//        }
//        for(int j=1;j<matrix[0].size();j++){
//            sum[0][j] += sum[0][j-1];
//        }
//        for(int i=1;i<matrix.size();i++){
//            for(int j=1;j<matrix[0].size();j++){
//                sum[i][j] += sum[i-1][j]+sum[i][j-1]-sum[i-1][j-1];
//            }
//        }
//    }
//
//    int sumRegion(int row1, int col1, int row2, int col2) {
//        if(row1==0 && col1==0){
//            return sum[row2][col2];
//        }
//        else if(col1==0){
//            return sum[row2][col2] - sum[row1-1][col2];
//        }
//        else if(row1==0){
//            return sum[row2][col2] - sum[row2][col1-1];
//        }
//
//        return sum[row2][col2] - sum[row1-1][col2] -sum[row2][col1-1] +sum[row1-1][col1-1];
//    }
//};


//338. 比特位计数
//给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
//int main(){
//    int num;
//    vector<int > ans(num+1,0);
//    for(int i=0;i<num+1;i++){
//        ans[i]=ans[i>>1]+(i&1);
//    }
//    return ans;
//}


//std::string to_string(int a){
//    stringstream ss;
//    ss << a;
//    std::string ans;
//    ss>>ans;
//    return ans;
//}
//
//class NACKStringBuilder
//{
//public:
//    NACKStringBuilder();
//    ~NACKStringBuilder();
//
//    void PushNACK(uint16_t nack);
//    std::string GetResult();
//
//private:
//    priority_queue <int,vector<int>,greater<int> > NackQueue;
//};
//
//NACKStringBuilder::NACKStringBuilder() {}
//
//NACKStringBuilder::~NACKStringBuilder() {}
//
//void NACKStringBuilder::PushNACK(uint16_t nack) {
//    NackQueue.push(nack);
//}
//
//
//
//std::string NACKStringBuilder::GetResult() {
//    int tmp;
//    std::string ansString="";
//    int flag=0;
//    if(!NackQueue.empty()){
//        tmp=NackQueue.top();
//        NackQueue.pop();
//        ansString += to_string(tmp);
//        while(!NackQueue.empty()){
//            if(NackQueue.top()!=tmp+1){
//                if(flag){
//                    ansString+='-';
//                    ansString+=to_string(tmp);
//                }
//                ansString+=',';
//                tmp=NackQueue.top();
//                ansString+=to_string(NackQueue.top());
//                NackQueue.pop();
//                flag=0;
//            }else{
//                flag = 1
//                tmp=NackQueue.top();
//                NackQueue.pop();
//            }
//        }
//        if(flag){
//            ansString+='-';
//            ansString+=to_string(tmp);
//        }
//    }
//    return ansString;
//}
//int main(){
//    NACKStringBuilder builder;
//    builder.PushNACK(5);
//    builder.PushNACK(7);
//    builder.PushNACK(9);
//    builder.PushNACK(10);
//    builder.PushNACK(11);
//    builder.PushNACK(12);
//    builder.PushNACK(15);
//    builder.PushNACK(18);
//    builder.PushNACK(19);
//    cout<<builder.GetResult();
//}

//void rotationMatrix(vector<vector<int>>& matrix) {
//    int n = matrix.size();
//    for(int i = 0; i < n / 2; i++)
//        matrix[i].swap(matrix[n - 1 - i]);
//    for(int i = 0; i < n; i++) {
//        for(int j = i; j < n; j++) {
//            swap(matrix[i][j], matrix[j][i]);
//        }
//    }
//}
//int main(){
//    int m=0,n=0;
//    cin>>m>>n;
//    vector<vector<int>> matrix{{1,2,3},{4,5,6},{7,8,9}};
//    rotationMatrix(matrix);
//    for(int j=0;j<m;j++){
//        for(int k=0;k<n;k++){
//            cout<<matrix[j][k]<<" ";
//        }
//        cout<<endl;
//    }
//}

//struct z{
//    int x;
//    int y;
//};
//
//int main(){
//    vector<vector<int>> matrix;
//    int m=matrix.size();
//    int n=matrix[0].size();
//    int a[m][n];
//    queue<z> po;
//    for(int i=0;i<m;i++)
//    {
//        a[i][0]=1;
//        a[i][n-1]=1;
//        z tmp;
//        tmp.x=i;
//        tmp.y=0;
//        po.push(tmp);
//        tmp.x=i;
//        tmp.y=n-1;
//        po.push(tmp);
//    }
//    for(int i=0;i<n;i++){
//        a[0][i]=1;
//        a[m-1][i]=1;
//        z tmp;
//        tmp.x=0;
//        tmp.y=i;
//        po.push(tmp);
//        tmp.x=m-1;
//        tmp.y=i;
//        po.push(tmp);
//    }
//    while(!po.empty()){
//        z tmp;
//        tmp=po.back();
//
//        if()
//    }
//
//}


/*
给双向有序（v从小到大排列）链表插入节点。
*/
//struct Node {
//    int v;
//    struct Node* prev;
//    struct Node* next;
//};
//
//int insert(struct Node** head,int v) {
//    struct Node* tmp = *head;
//    struct Node tmp2;
//    tmp2.v=v;
//    if(tmp->v>v){
//        tmp2.prev= nullptr;
//        tmp2.next=tmp;
//        tmp->prev=&tmp2;
//        *head = &tmp2;
//    }
//    while(!(tmp->v<v&&tmp->next->v>v)||tmp->next== nullptr){
//           tmp=tmp->next;
//    }
//    if(tmp->next!= nullptr){
//        tmp2.prev=tmp;
//        tmp2.next=tmp->next;
//        tmp->next->prev=&tmp2;
//        tmp->next=&tmp2;
//    }else{
//        tmp2.next= nullptr;
//        tmp2.prev=tmp;
//        tmp->next=&tmp2;
//    }
//    return v;
//};
//
//int search(int* array ,int count,int v){
//
//}


//int main(){
//    map<int ,int> a;
//    string q;
//    cin>>q;
//    if(q==" "){
//        cout<<"";
//        return 0;
//    }
//    for(int i=0;i<q.size();i++){
//        if(a[q[i]] == 2){
//            continue;
//        }else if(a[q[i]]==1){
//            a[q[i]]++;
//        }else{
//            a[q[i]]=1;
//        }
//    }
//    int i=0;
//    for(i=0;i<q.size();i++){
//        if(a[q[i]]==1){
//            cout<<q[i];
//            break;
//        }
//    }
//    if(i == q.size())
//        cout<<"";
//}

//int main(){
//    int n;
//    cin>>n;
//    vector<int> a(n);
//    for(int i=0;i<n;i++){
//        cin>>a[i];
//    }
//    sort(a.begin(),a.end());
//    int ans=0;
//    for(int i=1;i<n;i++){
//        ans+=a[i];
//    }
//    if(n%2==1)
//        ans+=a[0];
//    cout<<ans;
//}


//int main()
//{
//    int i, n, num[1000], a, b, c, d;
//    while(cin>>n)
//    {
//        for(i=0; i<n; i++) {
//            cin >> num[i];
//        }
//        sort(num, num + n);
//        int ans = 0;
//        while(n >= 4)
//        {
//            a = num[0];
//            b = num[1];
//            c = num[n - 2];
//            d = num[n - 1];
//            if(2 * b < a + c)
//            {
//                ans+=2*b+a+d;
//                n-=2;
//            }
//            else
//            {
//                ans+=a*2+c+d;
//                n-=2;
//            }
//        }
//        if(n == 3) {
//            ans+=num[0]+num[1]+num[2];
//        }
//        else if (n<=2) {
//            ans+=num[n-1];
//        }
//        cout<<ans;
//    }
//    return 0;
//}

//int main(){
//    vector<int> encoded={3,1};
//    int a=0;
//    for(int i=1;i<=encoded.size()+1;i++){
//        a=a^i;
//    }
//    for(int i=0;i<encoded.size();i+=2){
//        a=a^encoded[i];
//    }
//    vector<int> code(encoded.size()+1,0);
//    code[encoded.size()] = a;
//    for(int i=encoded.size()-1;i>=0;i--){
//        code[i]=encoded[i]^code[i+1];
//    }
//    for(int i=0;i<code.size();i++){
//        cout<<code[i]<<" ";
//    }
//}

//int main(){
//    vector<int> arr={1,3,4,8};
//    vector<vector<int>> queries={{0,1},{1,2},{0,3},{3,3}};
//
//    vector<int> a(arr.size());
//    a[0]=0^arr[0];
//    for(int i=1;i<arr.size();i++){
//        a[i]=a[i-1]^arr[i];
//    }
//    vector<int> ans;
//    for(int i=0;i<queries.size();i++){
//        if(queries[i][0]>=1)
//            ans.push_back(a[queries[i][1]]^a[queries[i][0]-1]);
//        else
//            ans.push_back(a[queries[i][1]]);
//    }
//    for(int i=0;i<queries.size();i++)
//        cout<<ans[i];
//}

//struct z{
//    int be;
//    int ed;
//    int jz;
//};
//bool cmd(z a1,z a2){
//    return a1.ed>a2.ed;
//}
//
//int dfs(int n,z a[],int index){
//    vector<int> ans;
//    for(int i=index+1;i<n;i++){
//           if(a[i].ed<=a[index].be){
//                ans.push_back(dfs(n,a,i));
//           }
//       }
//    sort(ans.begin(),ans.end());
//    if(ans.size()!=0)
//        return ans[ans.size()-1]+a[index].jz;
//    else
//        return a[index].jz;
//}
//
//int main(){
//    int n;
//    cin>>n;
//    z a[n];
//    for(int j=0;j<n;j++){
//        cin>>a[j].be;
//    }
//    for(int j=0;j<n;j++){
//        cin>>a[j].ed;
//    }
//    for(int j=0;j<n;j++){
//        cin>>a[j].jz;
//    }
//    sort(a,a+n,cmd);
//    vector<int> ans;
//    for(int i=0;i<n;i++){
//        ans.push_back(dfs(n,a,i));
//    }
//    sort(ans.begin(),ans.end());
//    cout<<ans[ans.size()-1];
//}




///*请完成下面这个函数，实现题目要求的功能
//当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
//******************************开始写代码******************************/
//const int N = 1005;
//int res;
//bool vis[N];
//int he[N];
//void dfs(int k, int index, vector<int>& scores, vector<int>& cards, int sum){
//    if(k == cards.size()){
//        res = max(res, sum);
//        return;
//    }
//    for(int i = 0; i < cards.size(); i ++){
//        if(vis[i] || index + cards[i] >= scores.size()) continue;
//        vis[i] = true;
//        if(res>sum+he[scores.size()]-he[index+1])
//            break;
//        dfs(k + 1, index + cards[i], scores, cards, sum + scores[index + cards[i]]);
//        res = max(res, sum + scores[cards[i] + index]);
//        vis[i] = false;
//    }
//}
//
//int procee(vector <int> scores, vector <int> cards) {
//
//    he[0]=scores[0];
//    for(int i=1;i<scores.size();i++){
//        he[i]=he[i-1]+scores[i];
//    }
//    res = scores[0];
//    dfs(0, 0, scores, cards, scores[0]);
//
//    return res;
//}
///******************************结束写代码******************************/
//
//
//int main() {
//    int res;
//
//    int _scores_size = 0;
//    cin >> _scores_size;
//    cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
//    vector<int> _scores;
//    int _scores_item;
//    for(int _scores_i=0; _scores_i<_scores_size; _scores_i++) {
//        cin >> _scores_item;
//        cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
//        _scores.push_back(_scores_item);
//    }
//
//
//
//    int _cards_size = 0;
//    cin >> _cards_size;
//    cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
//    vector<int> _cards;
//    int _cards_item;
//    for(int _cards_i=0; _cards_i<_cards_size; _cards_i++) {
//        cin >> _cards_item;
//        cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
//        _cards.push_back(_cards_item);
//    }
//    res = procee(_scores, _cards);
//    cout << res << endl;
//
//    return 0;
//
//}

//// 排序算法bobobo
//// 选择排序
//template<typename T>
//void selectionSort(vector<T> &a) {
//    if (a.size() == 0)
//        return;
//    for (int i = 0; i < a.size(); i++) {
//        int min = i;
//        for (int j = i + 1; j < a.size(); j++) {
//            if (a[j] < a[min]) {
//                min = j;
//            }
//        }
//        swap(a[min], a[i]);
//    }
//}
//
////插入排序
//template<typename T>
//void insertionSort(vector<T> &a) {
//    for (int i = 1; i < a.size(); i++) {
//        T temp = a[i];
//        int j;
//        for (j = i - 1; j > 0 && temp < a[j]; j--) {
//            a[j + 1] = a[j];
//        }
//        a[j] = temp;
//    }
//}
//
//template<typename T>
//void insertionSort(vector<T> &a, int l, int r) {
//    for (int i = l + 1; i <= r; i++) {
//        T temp = a[i];
//        int j;
//        for (j = i - 1; j > 0 && temp < a[j]; j--) {
//            a[j + 1] = a[j];
//        }
//        a[j] = temp;
//    }
//}
//
////归并排序
//template<typename T>
//void __merge(vector<T> &a, int l, int mid, int r) {
//    vector<T> aux(r - l + 1);
//    for (int i = l; i <= r; i++) {
//        aux[i - l] = a[i];
//    }
//    int i = l;
//    int j = mid + 1;
//    for (int k = l; k <= r; k++) {
//        if (i > mid) {
//            a[k] = aux[j - l];
//            j++;
//            continue;
//        } else if (j > r) {
//            a[k] = aux[i - l];
//            i++;
//            continue;
//        }
//        if (aux[i - l] < aux[j - l]) {
//            a[k] = aux[i - l];
//            i++;
//        } else {
//            a[k] = aux[j - l];
//            j++;
//        }
//    }
//
//}
//
//template<typename T>
//void __mergeSort(vector<T> &a, int l, int r) {
//    if (l >= r) {
//        return;
//    }
////    if(r - l <=15){
////        insertionSort(a,l,r);
////        return;
////    }
//    int mid = l / 2 + r / 2;
//    __mergeSort(a, l, mid);
//    __mergeSort(a, mid + 1, r);
//    if (a[mid] > a[mid + 1])
//        __merge(a, l, mid, r);
//}
//
//template<typename T>
//void mergeSort(vector<T> &a) {
//    __mergeSort(a, 0, a.size() - 1);
//}
//
//template<typename T>
//void mergeSortBU(vector<T> &a) {
//    for (int sz = 1; sz <= a.size() - 1; sz += sz) {
//        for (int i = 0; (i + sz) < (a.size() - 1); i += sz + sz)
//            __merge(a, i, i + sz - 1, (i + 2 * sz - 1) < (a.size() - 1) ? (i + 2 * sz - 1) : (a.size() - 1));
//    }
//}
//
////快速排序
//template<typename T>
//int __partition(vector<T> &a, int l, int r) {
//    swap(a[l], a[rand() % (r - l + 1) + l]);
//    int j = l;
//    for (int i = l + 1; i <= r; i++) {
//        if (a[i] < a[l]) {
//            swap(a[++j], a[i]);
//        }
//    }
//    swap(a[l], a[j]);
//    return j;
//}
//
//template<typename T>
//void __quickSort(vector<T> &a, int l, int r) {
////    if( l >= r)
////        return ;
//    if (r - l <= 15) {
//        insertionSort(a, l, r);
//        return;
//    }
//    int p = __partition(a, l, r);
//    __quickSort(a, l, p - 1);
//    __quickSort(a, p + 1, r);
//}
//
//template<typename T>
//void quickSort(vector<T> &a) {
//    srand(time(NULL));
//    __quickSort(a, 0, a.size() - 1);
//}
//
//template<typename T>
//int __partition2(vector<T> &a, int l, int r) {
//    swap(a[l], a[rand() % (r - l + 1) + l]);
//    int i = l + 1, j = r;
//    while (true) {
//        while (i <= r && a[i] < a[l]) i++;
//        while (j >= l + 1 && a[j] > a[l]) j--;
//        if (i > j) break;
//        swap(a[i], a[j]);
//        i++;
//        j--;
//    }
//    swap(a[l], a[j]);
//    return j;
//}
//
//template<typename T>
//void __quickSort2(vector<T> &a, int l, int r) {
////    if( l >= r)
////        return ;
//    if (r - l <= 15) {
//        insertionSort(a, l, r);
//        return;
//    }
//    int p = __partition2(a, l, r);
//    __quickSort2(a, l, p - 1);
//    __quickSort2(a, p + 1, r);
//}
//
//template<typename T>
//void quickSort2(vector<T> &a) {
//    srand(time(NULL));
//    __quickSort2(a, 0, a.size() - 1);
//}
//
//template<typename T>
//void __quickSort3Ways(vector<T> &a, int l, int r) {
////    if( l >= r)
////        return ;
//    if (r - l <= 15) {
//        insertionSort(a, l, r);
//        return;
//    }
//
//    swap(a[l], a[rand() % (r - l + 1) + l]);
//    int lt=l;
//    int gt=r+1;
//    int i=l+1;
//
//    while(i<gt){
//        if(a[i]<a[l]) swap(a[++lt],a[i++]);
//        else if(a[i]>a[l]) swap(a[--gt],a[i]);
//        else i++;
//    }
//    swap(a[l],a[lt]);
//
//    __quickSort3Ways(a, l, lt-1);
//    __quickSort3Ways(a, gt, r);
//}
//
//template<typename T>
//void quickSort3Ways(vector<T> &a) {
//    srand(time(NULL));
//    __quickSort3Ways(a, 0, a.size() - 1);
//}
//
//class Student {
//public:
//    string name;
//    int score;
//
//    Student(string name, int score) {
//        this->score = score;
//        this->name = name;
//    }
//
//    bool operator<(const Student &otherStudent) const {
//        return score < otherStudent.score;
//    }
//
//    friend ostream &operator<<(ostream &os, const Student &student) {
//        os << "Student: " << student.name << " " << student.score << endl;
//        return os;
//    }
//
//};
//
//
//int main() {
//    vector<int> a = {6, 5, 4, 3, 2, 1};
//    vector<double> b = {6.6, 5.5, 4.4, 3.3, 2.2, 1.1};
//    vector<Student> student = {{"abc", 6},
//                               {"bcd", 5},
//                               {"bcd", 4},
//                               {"bcd", 3},
//                               {"bcd", 2},
//                               {"bcd", 1},
//                               {"bcd", 0}};
//    vector<int> e = SortTestHelper::generateRandomArray<int>(100000, 0, 161124);
////    vector<int> d = {c.begin(), c.end()};
////    vector<int> e = SortTestHelper::generateNearlyOrderedArray<int>(50000, 10);
//    vector<int> f = {e.begin(), e.end()};
//    vector<int> j = {e.begin(), e.end()};
//    // 选择排序
////    SortTestHelper::testSort("selectionSort", selectionSort, e);
//    // 插入排序
////    SortTestHelper::testSort("insertionSort", insertionSort, f);
//    // 归并排序
////    SortTestHelper::testSort("mergeSort",mergeSort,f);
////    SortTestHelper::testSort("mergeSortBU",mergeSortBU,j);
//    // 快速排序
////    SortTestHelper::testSort("quickSort", quickSort, j);
////    SortTestHelper::testSort("quickSort2", quickSort2, f);
//    SortTestHelper::testSort("quickSort3", quickSort3Ways, e);
//}

template<typename T>
void __shiftDown(T arr[],int n,int k) {
    while (k * 2 +1 <= n) {
        int j = 2 * k;
        if (j + 1 <= n && arr[j + 1] > arr[j]) {
            j += 1;
        }
        if (arr[k] >= arr[j])
            break;
        swap(arr[k], arr[j]);
        k = j;
    }
}

template<typename T>
void heapSort1(T arr[],int n){
    MaxHeap<T> maxHeap = MaxHeap<T>(n);
    for(int i=0;i<n;i++){
        maxHeap.insert(arr[i]);
    }
    for(int i=n-1;i>=0;i--){
        arr[i] = maxHeap.extractMax();
    }
}

template<typename T>
void heapSort2(T arr[],int n){
    MaxHeap<T> maxHeap = MaxHeap<T>(arr,n);
    for(int i=n-1;i>=0;i--){
        arr[i] = maxHeap.extractMax();
    }
}

template<typename T>
void heapSort(T arr[],int n){
    // 从(最后一个元素的索引-1)/2开始
    // 最后一个元素的索引 = n-1
    for(int i = (n-1-1)/2;i>=0;i--){
        __shiftDown(arr,n,i);
    }
    for( int i = n-1;i>0;i--){
        swap(arr[0],arr[i]);
        __shiftDown(arr,i,0);
    }
}

int main() {
//    MaxHeap<int> maxHeap = MaxHeap<int>(20);
//    cout << maxHeap.size();
//    srand(time(NULL));
//    for (int i = 0; i < 30; i++) {
//        maxHeap.insert(i);
//    }
//    cout << maxHeap.size() << endl;
//    maxHeap.print();
//    cout << endl;
//    maxHeap.testPrint();
//    for(int k=0;k<15;k++){
//        cout<<maxHeap.extractMax()<<endl;
//    }
//    cout << maxHeap.size() << endl;
//    maxHeap.testPrint();
    vector<int> e = SortTestHelper::generateRandomArray<int>(100000, 0, 161124);
}
