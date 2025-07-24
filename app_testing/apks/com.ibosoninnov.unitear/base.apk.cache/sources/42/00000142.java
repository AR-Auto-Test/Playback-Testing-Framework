package b.b.c;

import android.content.Context;
import android.content.res.Configuration;
import android.os.Bundle;
import android.view.MenuInflater;
import android.view.View;
import android.view.ViewGroup;
import java.lang.ref.WeakReference;
import java.util.Iterator;

/* compiled from: AppCompatDelegate.java */
/* loaded from: classes.dex */
public abstract class j {

    /* renamed from: b  reason: collision with root package name */
    public static final b.f.c<WeakReference<j>> f565b = new b.f.c<>(0);

    /* renamed from: c  reason: collision with root package name */
    public static final Object f566c = new Object();

    public static void q(j jVar) {
        synchronized (f566c) {
            Iterator<WeakReference<j>> it = f565b.iterator();
            while (it.hasNext()) {
                j jVar2 = it.next().get();
                if (jVar2 == jVar || jVar2 == null) {
                    it.remove();
                }
            }
        }
    }

    public abstract void a(View view, ViewGroup.LayoutParams layoutParams);

    public Context b(Context context) {
        return context;
    }

    public abstract <T extends View> T c(int i);

    public int d() {
        return -100;
    }

    public abstract MenuInflater e();

    public abstract a f();

    public abstract void g();

    public abstract void h();

    public abstract void i(Configuration configuration);

    public abstract void j(Bundle bundle);

    public abstract void k();

    public abstract void l(Bundle bundle);

    public abstract void m();

    public abstract void n(Bundle bundle);

    public abstract void o();

    public abstract void p();

    public abstract boolean r(int i);

    public abstract void s(int i);

    public abstract void t(View view);

    public abstract void u(View view, ViewGroup.LayoutParams layoutParams);

    public void v(int i) {
    }

    public abstract void w(CharSequence charSequence);
}