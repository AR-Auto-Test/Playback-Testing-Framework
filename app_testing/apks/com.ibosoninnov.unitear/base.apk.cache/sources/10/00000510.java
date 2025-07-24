package b.q.b;

import android.view.View;
import androidx.fragment.app.Fragment;
import java.util.ArrayList;

/* compiled from: FragmentTransition.java */
/* loaded from: classes.dex */
public final class c0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Object f2411b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ k0 f2412c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ View f2413d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ Fragment f2414e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2415f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2416g;

    /* renamed from: h  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2417h;
    public final /* synthetic */ Object i;

    public c0(Object obj, k0 k0Var, View view, Fragment fragment, ArrayList arrayList, ArrayList arrayList2, ArrayList arrayList3, Object obj2) {
        this.f2411b = obj;
        this.f2412c = k0Var;
        this.f2413d = view;
        this.f2414e = fragment;
        this.f2415f = arrayList;
        this.f2416g = arrayList2;
        this.f2417h = arrayList3;
        this.i = obj2;
    }

    @Override // java.lang.Runnable
    public void run() {
        Object obj = this.f2411b;
        if (obj != null) {
            this.f2412c.n(obj, this.f2413d);
            this.f2416g.addAll(f0.h(this.f2412c, this.f2411b, this.f2414e, this.f2415f, this.f2413d));
        }
        if (this.f2417h != null) {
            if (this.i != null) {
                ArrayList<View> arrayList = new ArrayList<>();
                arrayList.add(this.f2413d);
                this.f2412c.o(this.i, this.f2417h, arrayList);
            }
            this.f2417h.clear();
            this.f2417h.add(this.f2413d);
        }
    }
}