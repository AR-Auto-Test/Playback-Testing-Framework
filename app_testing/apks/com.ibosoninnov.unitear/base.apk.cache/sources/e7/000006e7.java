package c.c.a;

import android.content.Context;
import android.content.ContextWrapper;
import c.c.a.b;
import c.c.a.m.v.l;
import java.util.List;
import java.util.Map;

/* compiled from: GlideContext.java */
/* loaded from: classes.dex */
public class d extends ContextWrapper {

    /* renamed from: a  reason: collision with root package name */
    public static final j<?, ?> f3425a = new a();

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f3426b;

    /* renamed from: c  reason: collision with root package name */
    public final g f3427c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.q.j.f f3428d;

    /* renamed from: e  reason: collision with root package name */
    public final b.a f3429e;

    /* renamed from: f  reason: collision with root package name */
    public final List<c.c.a.q.e<Object>> f3430f;

    /* renamed from: g  reason: collision with root package name */
    public final Map<Class<?>, j<?, ?>> f3431g;

    /* renamed from: h  reason: collision with root package name */
    public final l f3432h;
    public final e i;
    public final int j;
    public c.c.a.q.f k;

    public d(Context context, c.c.a.m.v.c0.b bVar, g gVar, c.c.a.q.j.f fVar, b.a aVar, Map<Class<?>, j<?, ?>> map, List<c.c.a.q.e<Object>> list, l lVar, e eVar, int i) {
        super(context.getApplicationContext());
        this.f3426b = bVar;
        this.f3427c = gVar;
        this.f3428d = fVar;
        this.f3429e = aVar;
        this.f3430f = list;
        this.f3431g = map;
        this.f3432h = lVar;
        this.i = eVar;
        this.j = i;
    }
}