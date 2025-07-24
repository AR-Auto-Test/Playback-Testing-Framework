package b.j.b;

import android.app.Activity;
import android.text.TextUtils;
import java.util.Arrays;

/* compiled from: ActivityCompat.java */
/* loaded from: classes.dex */
public class a extends b.j.c.a {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f2030b = 0;

    /* compiled from: ActivityCompat.java */
    /* renamed from: b.j.b.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0031a {
        void a(int i);
    }

    public static void c(Activity activity, String[] strArr, int i) {
        for (String str : strArr) {
            if (TextUtils.isEmpty(str)) {
                throw new IllegalArgumentException(c.b.a.a.a.v(c.b.a.a.a.x("Permission request for permissions "), Arrays.toString(strArr), " must not contain null or empty values"));
            }
        }
        if (activity instanceof InterfaceC0031a) {
            ((InterfaceC0031a) activity).a(i);
        }
        activity.requestPermissions(strArr, i);
    }
}