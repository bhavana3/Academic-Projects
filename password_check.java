/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rev;

/**
 *
 * @author bhavana
 */
public class password_check {
    
    
     public static void method_strengthen_passwords(String[] passwords) {
        String[] passwords1 = new String[passwords.length];
        for(int i = 0; i < passwords.length; i++) {
            String pwd1 = passwords[i];
            pwd1 = pwd1.replaceAll("[sS]", "5");
            
           if(pwd1.length() == 1) continue;
           
            if(pwd1.length() % 2 != 0) {
                int index = pwd1.length() / 2;
                if(Character.isDigit(pwd1.charAt(index))) {
                    String number = "" + pwd1.charAt(index);
                    int number_1 = 2 + Integer.parseInt(number);
                    String new_password =  pwd1.substring(0,index) + Integer.toString(number_1) + pwd1.substring(index+1,pwd1.length());
                    passwords1[i] = new_password;
                }
                else {
                    passwords1[i] = pwd1;
                }
            }
            
            else {
                char firstletter = pwd1.charAt(0);
                char lastletter = pwd1.charAt(pwd1.length()-1);
                if(Character.isUpperCase(firstletter)) {
                    firstletter = Character.toLowerCase(firstletter);
                }
                else  {
                    firstletter = Character.toUpperCase(firstletter);
                }

                 if(Character.isUpperCase(lastletter)) {
                    lastletter = Character.toLowerCase(lastletter);
                }
                else { 
                     lastletter = Character.toUpperCase(lastletter);
                }
                String new_password =  lastletter + pwd1.substring(1,pwd1.length()-1) + firstletter;
                pwd1 = new_password;
                passwords1[i] = new_password;
            } 
            System.out.println(passwords1[i]);
        }
    }
    
    public static void main(String[] args) {
        String[] passwords = {"Intel1gent", "DogCat", "thebeginneR", "FiveeSThree"};
        method_strengthen_passwords(passwords);
    }
    
}
