package c.c.a.m.v;

import android.util.Log;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/* compiled from: GlideException.java */
/* loaded from: classes.dex */
public final class r extends Exception {

    /* renamed from: b  reason: collision with root package name */
    public static final StackTraceElement[] f3787b = new StackTraceElement[0];

    /* renamed from: c  reason: collision with root package name */
    public final List<Throwable> f3788c;

    /* renamed from: d  reason: collision with root package name */
    public c.c.a.m.m f3789d;

    /* renamed from: e  reason: collision with root package name */
    public c.c.a.m.a f3790e;

    /* renamed from: f  reason: collision with root package name */
    public Class<?> f3791f;

    /* renamed from: g  reason: collision with root package name */
    public String f3792g;

    public r(String str) {
        List<Throwable> emptyList = Collections.emptyList();
        this.f3792g = str;
        setStackTrace(f3787b);
        this.f3788c = emptyList;
    }

    public static void b(List<Throwable> list, Appendable appendable) {
        try {
            c(list, appendable);
        } catch (IOException e2) {
            throw new RuntimeException(e2);
        }
    }

    public static void c(List<Throwable> list, Appendable appendable) {
        int size = list.size();
        int i = 0;
        while (i < size) {
            int i2 = i + 1;
            appendable.append("Cause (").append(String.valueOf(i2)).append(" of ").append(String.valueOf(size)).append("): ");
            Throwable th = list.get(i);
            if (th instanceof r) {
                ((r) th).f(appendable);
            } else {
                d(th, appendable);
            }
            i = i2;
        }
    }

    public static void d(Throwable th, Appendable appendable) {
        try {
            appendable.append(th.getClass().toString()).append(": ").append(th.getMessage()).append('\n');
        } catch (IOException unused) {
            throw new RuntimeException(th);
        }
    }

    public final void a(Throwable th, List<Throwable> list) {
        if (th instanceof r) {
            for (Throwable th2 : ((r) th).f3788c) {
                a(th2, list);
            }
            return;
        }
        list.add(th);
    }

    public void e(String str) {
        ArrayList arrayList = new ArrayList();
        a(this, arrayList);
        int size = arrayList.size();
        int i = 0;
        while (i < size) {
            StringBuilder x = c.b.a.a.a.x("Root cause (");
            int i2 = i + 1;
            x.append(i2);
            x.append(" of ");
            x.append(size);
            x.append(")");
            Log.i(str, x.toString(), (Throwable) arrayList.get(i));
            i = i2;
        }
    }

    public final void f(Appendable appendable) {
        d(this, appendable);
        b(this.f3788c, new a(appendable));
    }

    @Override // java.lang.Throwable
    public Throwable fillInStackTrace() {
        return this;
    }

    @Override // java.lang.Throwable
    public String getMessage() {
        String str;
        String str2;
        StringBuilder sb = new StringBuilder(71);
        sb.append(this.f3792g);
        String str3 = "";
        if (this.f3791f != null) {
            StringBuilder x = c.b.a.a.a.x(", ");
            x.append(this.f3791f);
            str = x.toString();
        } else {
            str = "";
        }
        sb.append(str);
        if (this.f3790e != null) {
            StringBuilder x2 = c.b.a.a.a.x(", ");
            x2.append(this.f3790e);
            str2 = x2.toString();
        } else {
            str2 = "";
        }
        sb.append(str2);
        if (this.f3789d != null) {
            StringBuilder x3 = c.b.a.a.a.x(", ");
            x3.append(this.f3789d);
            str3 = x3.toString();
        }
        sb.append(str3);
        ArrayList arrayList = new ArrayList();
        a(this, arrayList);
        if (arrayList.isEmpty()) {
            return sb.toString();
        }
        if (arrayList.size() == 1) {
            sb.append("\nThere was 1 root cause:");
        } else {
            sb.append("\nThere were ");
            sb.append(arrayList.size());
            sb.append(" root causes:");
        }
        Iterator it = arrayList.iterator();
        while (it.hasNext()) {
            Throwable th = (Throwable) it.next();
            sb.append('\n');
            sb.append(th.getClass().getName());
            sb.append('(');
            sb.append(th.getMessage());
            sb.append(')');
        }
        sb.append("\n call GlideException#logRootCauses(String) for more detail");
        return sb.toString();
    }

    @Override // java.lang.Throwable
    public void printStackTrace() {
        f(System.err);
    }

    @Override // java.lang.Throwable
    public void printStackTrace(PrintStream printStream) {
        f(printStream);
    }

    @Override // java.lang.Throwable
    public void printStackTrace(PrintWriter printWriter) {
        f(printWriter);
    }

    /* compiled from: GlideException.java */
    /* loaded from: classes.dex */
    public static final class a implements Appendable {

        /* renamed from: b  reason: collision with root package name */
        public final Appendable f3793b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f3794c = true;

        public a(Appendable appendable) {
            this.f3793b = appendable;
        }

        @Override // java.lang.Appendable
        public Appendable append(char c2) {
            if (this.f3794c) {
                this.f3794c = false;
                this.f3793b.append("  ");
            }
            this.f3794c = c2 == '\n';
            this.f3793b.append(c2);
            return this;
        }

        @Override // java.lang.Appendable
        public Appendable append(CharSequence charSequence) {
            if (charSequence == null) {
                charSequence = "";
            }
            append(charSequence, 0, charSequence.length());
            return this;
        }

        @Override // java.lang.Appendable
        public Appendable append(CharSequence charSequence, int i, int i2) {
            if (charSequence == null) {
                charSequence = "";
            }
            boolean z = false;
            if (this.f3794c) {
                this.f3794c = false;
                this.f3793b.append("  ");
            }
            if (charSequence.length() > 0 && charSequence.charAt(i2 - 1) == '\n') {
                z = true;
            }
            this.f3794c = z;
            this.f3793b.append(charSequence, i, i2);
            return this;
        }
    }

    public r(String str, Throwable th) {
        List<Throwable> singletonList = Collections.singletonList(th);
        this.f3792g = str;
        setStackTrace(f3787b);
        this.f3788c = singletonList;
    }

    public r(String str, List<Throwable> list) {
        this.f3792g = str;
        setStackTrace(f3787b);
        this.f3788c = list;
    }
}