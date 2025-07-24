package b.q.b;

import androidx.fragment.app.Fragment;
import b.q.b.f0;
import b.q.b.q;

/* compiled from: FragmentTransition.java */
/* loaded from: classes.dex */
public final class b0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ f0.a f2405b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Fragment f2406c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ b.j.f.b f2407d;

    public b0(f0.a aVar, Fragment fragment, b.j.f.b bVar) {
        this.f2405b = aVar;
        this.f2406c = fragment;
        this.f2407d = bVar;
    }

    @Override // java.lang.Runnable
    public void run() {
        ((q.b) this.f2405b).a(this.f2406c, this.f2407d);
    }
}