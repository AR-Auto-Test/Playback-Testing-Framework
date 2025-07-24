package b.q.b;

import androidx.fragment.app.Fragment;
import b.q.b.f0;
import b.q.b.q;

/* compiled from: FragmentTransition.java */
/* loaded from: classes.dex */
public final class z implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ f0.a f2557b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Fragment f2558c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ b.j.f.b f2559d;

    public z(f0.a aVar, Fragment fragment, b.j.f.b bVar) {
        this.f2557b = aVar;
        this.f2558c = fragment;
        this.f2559d = bVar;
    }

    @Override // java.lang.Runnable
    public void run() {
        ((q.b) this.f2557b).a(this.f2558c, this.f2559d);
    }
}