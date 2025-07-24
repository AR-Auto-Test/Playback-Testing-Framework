package b.t;

/* compiled from: ViewModelProvider.java */
/* loaded from: classes.dex */
public abstract class v extends x implements u {
    public <T extends s> T a(Class<T> cls) {
        throw new UnsupportedOperationException("create(String, Class<?>) must be called on implementaions of KeyedFactory");
    }

    public abstract <T extends s> T c(String str, Class<T> cls);
}