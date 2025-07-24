package c.d.b.a.q;

import java.util.concurrent.CompletionException;
import java.util.function.Function;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class p implements Function {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ p f4348a = new p();

    @Override // java.util.function.Function
    public final Object apply(Object obj) {
        throw new CompletionException("Texture Load Error", (Throwable) obj);
    }
}