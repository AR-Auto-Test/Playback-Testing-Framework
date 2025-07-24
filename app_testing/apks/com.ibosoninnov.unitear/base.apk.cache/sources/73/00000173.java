package b.b.g.i;

import android.content.Context;
import android.view.LayoutInflater;
import b.b.g.i.m;

/* compiled from: BaseMenuPresenter.java */
/* loaded from: classes.dex */
public abstract class b implements m {

    /* renamed from: b  reason: collision with root package name */
    public Context f687b;

    /* renamed from: c  reason: collision with root package name */
    public Context f688c;

    /* renamed from: d  reason: collision with root package name */
    public g f689d;

    /* renamed from: e  reason: collision with root package name */
    public LayoutInflater f690e;

    /* renamed from: f  reason: collision with root package name */
    public m.a f691f;

    /* renamed from: g  reason: collision with root package name */
    public int f692g;

    /* renamed from: h  reason: collision with root package name */
    public int f693h;
    public n i;
    public int j;

    public b(Context context, int i, int i2) {
        this.f687b = context;
        this.f690e = LayoutInflater.from(context);
        this.f692g = i;
        this.f693h = i2;
    }

    @Override // b.b.g.i.m
    public boolean collapseItemActionView(g gVar, i iVar) {
        return false;
    }

    @Override // b.b.g.i.m
    public boolean expandItemActionView(g gVar, i iVar) {
        return false;
    }

    @Override // b.b.g.i.m
    public int getId() {
        return this.j;
    }

    @Override // b.b.g.i.m
    public void setCallback(m.a aVar) {
        this.f691f = aVar;
    }
}