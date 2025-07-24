package c.c.a.m.v;

/* compiled from: CallbackException.java */
/* loaded from: classes.dex */
public final class c extends RuntimeException {
    public c(Throwable th) {
        super("Unexpected exception thrown by non-Glide code", th);
    }
}