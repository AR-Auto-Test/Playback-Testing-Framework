package b.q.b;

import android.view.View;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FragmentTransitionImpl.java */
/* loaded from: classes.dex */
public class h0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f2469b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2470c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2471d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2472e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2473f;

    public h0(k0 k0Var, int i, ArrayList arrayList, ArrayList arrayList2, ArrayList arrayList3, ArrayList arrayList4) {
        this.f2469b = i;
        this.f2470c = arrayList;
        this.f2471d = arrayList2;
        this.f2472e = arrayList3;
        this.f2473f = arrayList4;
    }

    @Override // java.lang.Runnable
    public void run() {
        for (int i = 0; i < this.f2469b; i++) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            ((View) this.f2470c.get(i)).setTransitionName((String) this.f2471d.get(i));
            ((View) this.f2472e.get(i)).setTransitionName((String) this.f2473f.get(i));
        }
    }
}