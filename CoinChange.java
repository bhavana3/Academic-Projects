/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author bhavana
 */
public class CoinChange {
    
   
    public static int coin_chnage(int[] coins, int V, int m) {
        int[] total_coins = new int[V+1];
        
        total_coins[0] = 0;
        for(int i = 1; i <= V; i++)
            total_coins[i] = Integer.MAX_VALUE;
        
        for(int i= 1; i <= V; i++) {
            for(int j = 0; j < m; j++) {
                if (coins[j] <= i) {
                    int sub_total = total_coins[i-coins[j]];
                    if (sub_total != Integer.MAX_VALUE && (1 + sub_total) < total_coins[i]) {
                        total_coins[i] = 1 + sub_total;
                    }
                }
            }
        }
        
        return total_coins[V];
    }
    
    
    public static void main(String[] args) {
        int[] coins = {1,3,5,7};
        int total = 11;
        System.out.println(coin_chnage(coins, total, coins.length));
    }
}
