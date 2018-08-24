/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author bhavana
 */
public class CombineList {
    
    public static int[] merge_sorted_lists(int[] a, int[] b) {
        int[] mergedList = new int[a.length+b.length];
        
        int i = 0, j= 0, k = 0;
        while ( i < a.length && j < b.length) {
            if (a[i] <= b[j]) {
                mergedList[k] = a[i];
                i++;
            }
            else {
                mergedList[k] = b[j];
                j++;
            }
            k++;
        }
        //To handle cases where either the first or second array elements are still remaining
        while(i < a.length) {
            mergedList[k] = a[i];
            i++;
            k++;
        }
        
        while(j < b.length) {
            mergedList[k] = b[j];
            j++;
            k++;
        }
        return mergedList;
    }
    
    public static void main(String[] args) {
       int[] a = {3,5,6,7,8,10};
       int[] b = {1,2,4,9,11};
       int[] mergeList = merge_sorted_lists(a,b);
       
       for(int i = 0; i < mergeList.length; i++)
           System.out.println(mergeList[i]);
       
    }
    
}
