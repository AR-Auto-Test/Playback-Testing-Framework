package b.q.b;

import android.graphics.Rect;
import android.view.View;
import androidx.fragment.app.Fragment;
import b.q.b.f0;
import java.util.ArrayList;

/* compiled from: FragmentTransition.java */
/* loaded from: classes.dex */
public final class e0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ k0 f2429b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ b.f.a f2430c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Object f2431d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ f0.b f2432e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2433f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ View f2434g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ Fragment f2435h;
    public final /* synthetic */ Fragment i;
    public final /* synthetic */ boolean j;
    public final /* synthetic */ ArrayList k;
    public final /* synthetic */ Object l;
    public final /* synthetic */ Rect m;

    public e0(k0 k0Var, b.f.a aVar, Object obj, f0.b bVar, ArrayList arrayList, View view, Fragment fragment, Fragment fragment2, boolean z, ArrayList arrayList2, Object obj2, Rect rect) {
        this.f2429b = k0Var;
        this.f2430c = aVar;
        this.f2431d = obj;
        this.f2432e = bVar;
        this.f2433f = arrayList;
        this.f2434g = view;
        this.f2435h = fragment;
        this.i = fragment2;
        this.j = z;
        this.k = arrayList2;
        this.l = obj2;
        this.m = rect;
    }

    @Override // java.lang.Runnable
    public void run() {
        b.f.a<String, View> e2 = f0.e(this.f2429b, this.f2430c, this.f2431d, this.f2432e);
        if (e2 != null) {
            this.f2433f.addAll(e2.values());
            this.f2433f.add(this.f2434g);
        }
        f0.c(this.f2435h, this.i, this.j, e2, false);
        Object obj = this.f2431d;
        if (obj != null) {
            this.f2429b.v(obj, this.k, this.f2433f);
            View k = f0.k(e2, this.f2432e, this.l, this.j);
            if (k != null) {
                this.f2429b.j(k, this.m);
            }
        }
    }
}