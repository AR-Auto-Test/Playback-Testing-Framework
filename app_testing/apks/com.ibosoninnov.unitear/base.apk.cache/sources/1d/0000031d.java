package b.d.b.d1.k1;

import android.os.Handler;
import android.os.Looper;
import b.j.b.d;

/* compiled from: MainThreadAsyncHandler.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static volatile Handler f1515a;

    public static Handler a() {
        if (f1515a != null) {
            return f1515a;
        }
        synchronized (a.class) {
            if (f1515a == null) {
                f1515a = d.q(Looper.getMainLooper());
            }
        }
        return f1515a;
    }
}