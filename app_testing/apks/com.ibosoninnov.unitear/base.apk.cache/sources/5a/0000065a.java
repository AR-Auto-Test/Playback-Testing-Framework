package c.a.a;

import android.content.Context;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.zip.ZipInputStream;

/* compiled from: LottieCompositionFactory.java */
/* loaded from: classes.dex */
public class g implements Callable<p<d>> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Context f3066b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f3067c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ String f3068d;

    public g(Context context, String str, String str2) {
        this.f3066b = context;
        this.f3067c = str;
        this.f3068d = str2;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // java.util.concurrent.Callable
    public p<d> call() {
        p<d> b2;
        Context context = this.f3066b;
        String str = this.f3067c;
        String str2 = this.f3068d;
        try {
            if (str.endsWith(".zip")) {
                b2 = e.d(new ZipInputStream(context.getAssets().open(str)), str2);
            } else {
                b2 = e.b(context.getAssets().open(str), str2);
            }
            return b2;
        } catch (IOException e2) {
            return new p<>(e2);
        }
    }
}