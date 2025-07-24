package c.e.b;

import android.util.Log;
import java.util.HashMap;

/* compiled from: YoutubeHelper.java */
/* loaded from: classes2.dex */
public class cf {
    public String a(String str) {
        String str2;
        if (str.contains("?t=")) {
            str = str.substring(str.lastIndexOf("?t="));
        }
        if (!str.contains("youtu")) {
            str = c.b.a.a.a.q("youtube.com/watch?v=", str);
        }
        String replace = str.trim().replace("youtu.be/", "youtube.com/watch?v=").replace("www.youtube", "youtube").replace("youtube.com/embed/", "youtube.com/watch?v=").replace("/watch#", "/watch?");
        if (replace.contains("?")) {
            replace = replace.substring(replace.indexOf(63) + 1);
        }
        HashMap hashMap = new HashMap();
        String[] split = replace.split("&");
        int length = split.length;
        int i = 0;
        while (true) {
            str2 = "";
            if (i >= length) {
                break;
            }
            String[] split2 = split[i].split("=");
            String str3 = split2[0];
            if (split2.length == 2) {
                str2 = split2[1].replace("\\", "");
            }
            hashMap.put(str3, str2);
            i++;
        }
        String q = c.b.a.a.a.q("https://youtube.com/watch?v=", hashMap.get("v") != null ? (String) hashMap.get("v") : "");
        if (q == null) {
            Log.e("YoutubeHelper", "ITS NOT A YOUTUBE URL");
        }
        return q.substring(q.lastIndexOf("?v=") + 3, q.length());
    }
}