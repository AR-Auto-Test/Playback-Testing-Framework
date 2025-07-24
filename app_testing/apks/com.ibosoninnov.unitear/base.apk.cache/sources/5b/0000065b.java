package c.a.a;

import android.content.Context;
import android.content.res.Resources;
import java.lang.ref.WeakReference;
import java.util.concurrent.Callable;

/* compiled from: LottieCompositionFactory.java */
/* loaded from: classes.dex */
public class h implements Callable<p<d>> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ WeakReference f3069b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Context f3070c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ int f3071d;

    public h(WeakReference weakReference, Context context, int i) {
        this.f3069b = weakReference;
        this.f3070c = context;
        this.f3071d = i;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // java.util.concurrent.Callable
    public p<d> call() {
        Context context = (Context) this.f3069b.get();
        if (context == null) {
            context = this.f3070c;
        }
        int i = this.f3071d;
        try {
            return e.b(context.getResources().openRawResource(i), e.f(context, i));
        } catch (Resources.NotFoundException e2) {
            return new p<>(e2);
        }
    }
}