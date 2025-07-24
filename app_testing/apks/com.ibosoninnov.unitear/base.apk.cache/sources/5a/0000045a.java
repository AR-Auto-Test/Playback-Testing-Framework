package b.j.g;

import android.content.Context;
import b.j.g.j;
import java.util.concurrent.Callable;

/* compiled from: FontRequestWorker.java */
/* loaded from: classes.dex */
public class f implements Callable<j.a> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ String f2134b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Context f2135c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ e f2136d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ int f2137e;

    public f(String str, Context context, e eVar, int i) {
        this.f2134b = str;
        this.f2135c = context;
        this.f2136d = eVar;
        this.f2137e = i;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // java.util.concurrent.Callable
    public j.a call() {
        return j.a(this.f2134b, this.f2135c, this.f2136d, this.f2137e);
    }
}