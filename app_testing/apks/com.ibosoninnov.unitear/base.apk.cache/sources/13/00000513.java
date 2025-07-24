package b.q.b;

import android.graphics.Rect;
import android.view.View;
import androidx.fragment.app.Fragment;

/* compiled from: FragmentTransition.java */
/* loaded from: classes.dex */
public final class d0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Fragment f2421b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Fragment f2422c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ boolean f2423d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ b.f.a f2424e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ View f2425f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ k0 f2426g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ Rect f2427h;

    public d0(Fragment fragment, Fragment fragment2, boolean z, b.f.a aVar, View view, k0 k0Var, Rect rect) {
        this.f2421b = fragment;
        this.f2422c = fragment2;
        this.f2423d = z;
        this.f2424e = aVar;
        this.f2425f = view;
        this.f2426g = k0Var;
        this.f2427h = rect;
    }

    @Override // java.lang.Runnable
    public void run() {
        f0.c(this.f2421b, this.f2422c, this.f2423d, this.f2424e, false);
        View view = this.f2425f;
        if (view != null) {
            this.f2426g.j(view, this.f2427h);
        }
    }
}