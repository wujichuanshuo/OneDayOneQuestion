#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <stack>
#include <cstdio>
#include <algorithm>

using namespace  std;

int gcd(int a,int b)
{
    return b>0?gcd(b,a%b):a;
}

int lcm(int a,int b){
    return a*b/gcd(a,b);
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
int main(){
    vector<int> A;
    int a=0,b=0;
    if(A.size()==1)
        return true;
    for(int i=1;i<A.size();i++){
        if(A[i-1]<A[i])a=1;
        if(A[i-1]>A[i])b=1;
    }
    if(a+b==2)return false;
    return true;
}