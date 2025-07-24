package b.j.g;

import android.content.Context;
import b.j.g.j;
import java.util.concurrent.Callable;

/* compiled from: FontRequestWorker.java */
/* loaded from: classes.dex */
public class h implements Callable<j.a> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ String f2139b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Context f2140c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ e f2141d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ int f2142e;

    public h(String str, Context context, e eVar, int i) {
        this.f2139b = str;
        this.f2140c = context;
        this.f2141d = eVar;
        this.f2142e = i;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // java.util.concurrent.Callable
    public j.a call() {
        return j.a(this.f2139b, this.f2140c, this.f2141d, this.f2142e);
    }
}