package c.a.a;

import java.io.InputStream;
import java.util.concurrent.Callable;

/* compiled from: LottieCompositionFactory.java */
/* loaded from: classes.dex */
public class i implements Callable<p<d>> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ InputStream f3072b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f3073c;

    public i(InputStream inputStream, String str) {
        this.f3072b = inputStream;
        this.f3073c = str;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // java.util.concurrent.Callable
    public p<d> call() {
        return e.b(this.f3072b, this.f3073c);
    }
}