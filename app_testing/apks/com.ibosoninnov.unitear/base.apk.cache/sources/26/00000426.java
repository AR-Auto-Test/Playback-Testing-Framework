package b.j.b;

import android.util.Log;
import java.lang.reflect.Method;

/* compiled from: ActivityRecreator.java */
/* loaded from: classes.dex */
public class c implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Object f2048b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Object f2049c;

    public c(Object obj, Object obj2) {
        this.f2048b = obj;
        this.f2049c = obj2;
    }

    @Override // java.lang.Runnable
    public void run() {
        try {
            Method method = b.f2034d;
            if (method != null) {
                method.invoke(this.f2048b, this.f2049c, Boolean.FALSE, "AppCompat recreation");
            } else {
                b.f2035e.invoke(this.f2048b, this.f2049c, Boolean.FALSE);
            }
        } catch (RuntimeException e2) {
            if (e2.getClass() == RuntimeException.class && e2.getMessage() != null && e2.getMessage().startsWith("Unable to stop")) {
                throw e2;
            }
        } catch (Throwable th) {
            Log.e("ActivityRecreator", "Exception while invoking performStopActivity", th);
        }
    }
}