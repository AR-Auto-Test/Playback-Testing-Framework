package b.g.a;

import com.google.common.util.concurrent.ListenableFuture;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReferenceFieldUpdater;
import java.util.concurrent.locks.LockSupport;
import java.util.logging.Level;
import java.util.logging.Logger;

/* compiled from: AbstractResolvableFuture.java */
/* loaded from: classes.dex */
public abstract class a<V> implements ListenableFuture<V> {

    /* renamed from: b  reason: collision with root package name */
    public static final boolean f1781b = Boolean.parseBoolean(System.getProperty("guava.concurrent.generate_cancellation_cause", "false"));

    /* renamed from: c  reason: collision with root package name */
    public static final Logger f1782c = Logger.getLogger(a.class.getName());

    /* renamed from: d  reason: collision with root package name */
    public static final b f1783d;

    /* renamed from: e  reason: collision with root package name */
    public static final Object f1784e;

    /* renamed from: f  reason: collision with root package name */
    public volatile Object f1785f;

    /* renamed from: g  reason: collision with root package name */
    public volatile e f1786g;

    /* renamed from: h  reason: collision with root package name */
    public volatile i f1787h;

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static abstract class b {
        public b(C0027a c0027a) {
        }

        public abstract boolean a(a<?> aVar, e eVar, e eVar2);

        public abstract boolean b(a<?> aVar, Object obj, Object obj2);

        public abstract boolean c(a<?> aVar, i iVar, i iVar2);

        public abstract void d(i iVar, i iVar2);

        public abstract void e(i iVar, Thread thread);
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class c {

        /* renamed from: a  reason: collision with root package name */
        public static final c f1788a;

        /* renamed from: b  reason: collision with root package name */
        public static final c f1789b;

        /* renamed from: c  reason: collision with root package name */
        public final boolean f1790c;

        /* renamed from: d  reason: collision with root package name */
        public final Throwable f1791d;

        static {
            if (a.f1781b) {
                f1789b = null;
                f1788a = null;
                return;
            }
            f1789b = new c(false, null);
            f1788a = new c(true, null);
        }

        public c(boolean z, Throwable th) {
            this.f1790c = z;
            this.f1791d = th;
        }
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class d {

        /* renamed from: a  reason: collision with root package name */
        public final Throwable f1792a;

        /* compiled from: AbstractResolvableFuture.java */
        /* renamed from: b.g.a.a$d$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public static class C0028a extends Throwable {
            public C0028a(String str) {
                super(str);
            }

            @Override // java.lang.Throwable
            public synchronized Throwable fillInStackTrace() {
                return this;
            }
        }

        static {
            C0028a c0028a = new C0028a("Failure occurred while trying to finish a future.");
            boolean z = a.f1781b;
            Objects.requireNonNull(c0028a);
        }

        public d(Throwable th) {
            boolean z = a.f1781b;
            Objects.requireNonNull(th);
            this.f1792a = th;
        }
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class e {

        /* renamed from: a  reason: collision with root package name */
        public static final e f1793a = new e(null, null);

        /* renamed from: b  reason: collision with root package name */
        public final Runnable f1794b;

        /* renamed from: c  reason: collision with root package name */
        public final Executor f1795c;

        /* renamed from: d  reason: collision with root package name */
        public e f1796d;

        public e(Runnable runnable, Executor executor) {
            this.f1794b = runnable;
            this.f1795c = executor;
        }
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class f extends b {

        /* renamed from: a  reason: collision with root package name */
        public final AtomicReferenceFieldUpdater<i, Thread> f1797a;

        /* renamed from: b  reason: collision with root package name */
        public final AtomicReferenceFieldUpdater<i, i> f1798b;

        /* renamed from: c  reason: collision with root package name */
        public final AtomicReferenceFieldUpdater<a, i> f1799c;

        /* renamed from: d  reason: collision with root package name */
        public final AtomicReferenceFieldUpdater<a, e> f1800d;

        /* renamed from: e  reason: collision with root package name */
        public final AtomicReferenceFieldUpdater<a, Object> f1801e;

        public f(AtomicReferenceFieldUpdater<i, Thread> atomicReferenceFieldUpdater, AtomicReferenceFieldUpdater<i, i> atomicReferenceFieldUpdater2, AtomicReferenceFieldUpdater<a, i> atomicReferenceFieldUpdater3, AtomicReferenceFieldUpdater<a, e> atomicReferenceFieldUpdater4, AtomicReferenceFieldUpdater<a, Object> atomicReferenceFieldUpdater5) {
            super(null);
            this.f1797a = atomicReferenceFieldUpdater;
            this.f1798b = atomicReferenceFieldUpdater2;
            this.f1799c = atomicReferenceFieldUpdater3;
            this.f1800d = atomicReferenceFieldUpdater4;
            this.f1801e = atomicReferenceFieldUpdater5;
        }

        @Override // b.g.a.a.b
        public boolean a(a<?> aVar, e eVar, e eVar2) {
            return this.f1800d.compareAndSet(aVar, eVar, eVar2);
        }

        @Override // b.g.a.a.b
        public boolean b(a<?> aVar, Object obj, Object obj2) {
            return this.f1801e.compareAndSet(aVar, obj, obj2);
        }

        @Override // b.g.a.a.b
        public boolean c(a<?> aVar, i iVar, i iVar2) {
            return this.f1799c.compareAndSet(aVar, iVar, iVar2);
        }

        @Override // b.g.a.a.b
        public void d(i iVar, i iVar2) {
            this.f1798b.lazySet(iVar, iVar2);
        }

        @Override // b.g.a.a.b
        public void e(i iVar, Thread thread) {
            this.f1797a.lazySet(iVar, thread);
        }
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class g<V> implements Runnable {
        @Override // java.lang.Runnable
        public void run() {
            throw null;
        }
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class h extends b {
        public h() {
            super(null);
        }

        @Override // b.g.a.a.b
        public boolean a(a<?> aVar, e eVar, e eVar2) {
            synchronized (aVar) {
                if (aVar.f1786g == eVar) {
                    aVar.f1786g = eVar2;
                    return true;
                }
                return false;
            }
        }

        @Override // b.g.a.a.b
        public boolean b(a<?> aVar, Object obj, Object obj2) {
            synchronized (aVar) {
                if (aVar.f1785f == obj) {
                    aVar.f1785f = obj2;
                    return true;
                }
                return false;
            }
        }

        @Override // b.g.a.a.b
        public boolean c(a<?> aVar, i iVar, i iVar2) {
            synchronized (aVar) {
                if (aVar.f1787h == iVar) {
                    aVar.f1787h = iVar2;
                    return true;
                }
                return false;
            }
        }

        @Override // b.g.a.a.b
        public void d(i iVar, i iVar2) {
            iVar.f1804c = iVar2;
        }

        @Override // b.g.a.a.b
        public void e(i iVar, Thread thread) {
            iVar.f1803b = thread;
        }
    }

    /* compiled from: AbstractResolvableFuture.java */
    /* loaded from: classes.dex */
    public static final class i {

        /* renamed from: a  reason: collision with root package name */
        public static final i f1802a = new i(false);

        /* renamed from: b  reason: collision with root package name */
        public volatile Thread f1803b;

        /* renamed from: c  reason: collision with root package name */
        public volatile i f1804c;

        public i(boolean z) {
        }

        public i() {
            a.f1783d.e(this, Thread.currentThread());
        }
    }

    static {
        b hVar;
        try {
            hVar = new f(AtomicReferenceFieldUpdater.newUpdater(i.class, Thread.class, "b"), AtomicReferenceFieldUpdater.newUpdater(i.class, i.class, "c"), AtomicReferenceFieldUpdater.newUpdater(a.class, i.class, "h"), AtomicReferenceFieldUpdater.newUpdater(a.class, e.class, "g"), AtomicReferenceFieldUpdater.newUpdater(a.class, Object.class, "f"));
            th = null;
        } catch (Throwable th) {
            th = th;
            hVar = new h();
        }
        f1783d = hVar;
        if (th != null) {
            f1782c.log(Level.SEVERE, "SafeAtomicHelper is broken!", th);
        }
        f1784e = new Object();
    }

    public static void b(a<?> aVar) {
        i iVar;
        e eVar;
        do {
            iVar = aVar.f1787h;
        } while (!f1783d.c(aVar, iVar, i.f1802a));
        while (iVar != null) {
            Thread thread = iVar.f1803b;
            if (thread != null) {
                iVar.f1803b = null;
                LockSupport.unpark(thread);
            }
            iVar = iVar.f1804c;
        }
        do {
            eVar = aVar.f1786g;
        } while (!f1783d.a(aVar, eVar, e.f1793a));
        e eVar2 = null;
        while (eVar != null) {
            e eVar3 = eVar.f1796d;
            eVar.f1796d = eVar2;
            eVar2 = eVar;
            eVar = eVar3;
        }
        while (eVar2 != null) {
            e eVar4 = eVar2.f1796d;
            Runnable runnable = eVar2.f1794b;
            if (!(runnable instanceof g)) {
                c(runnable, eVar2.f1795c);
                eVar2 = eVar4;
            } else {
                Objects.requireNonNull((g) runnable);
                throw null;
            }
        }
    }

    public static void c(Runnable runnable, Executor executor) {
        try {
            executor.execute(runnable);
        } catch (RuntimeException e2) {
            Logger logger = f1782c;
            Level level = Level.SEVERE;
            logger.log(level, "RuntimeException while executing runnable " + runnable + " with executor " + executor, (Throwable) e2);
        }
    }

    public static <V> V e(Future<V> future) {
        V v;
        boolean z = false;
        while (true) {
            try {
                v = future.get();
                break;
            } catch (InterruptedException unused) {
                z = true;
            } catch (Throwable th) {
                if (z) {
                    Thread.currentThread().interrupt();
                }
                throw th;
            }
        }
        if (z) {
            Thread.currentThread().interrupt();
        }
        return v;
    }

    public final void a(StringBuilder sb) {
        try {
            Object e2 = e(this);
            sb.append("SUCCESS, result=[");
            sb.append(e2 == this ? "this future" : String.valueOf(e2));
            sb.append("]");
        } catch (CancellationException unused) {
            sb.append("CANCELLED");
        } catch (RuntimeException e3) {
            sb.append("UNKNOWN, cause=[");
            sb.append(e3.getClass());
            sb.append(" thrown from get()]");
        } catch (ExecutionException e4) {
            sb.append("FAILURE, cause=[");
            sb.append(e4.getCause());
            sb.append("]");
        }
    }

    @Override // com.google.common.util.concurrent.ListenableFuture
    public final void addListener(Runnable runnable, Executor executor) {
        Objects.requireNonNull(runnable);
        Objects.requireNonNull(executor);
        e eVar = this.f1786g;
        if (eVar != e.f1793a) {
            e eVar2 = new e(runnable, executor);
            do {
                eVar2.f1796d = eVar;
                if (f1783d.a(this, eVar, eVar2)) {
                    return;
                }
                eVar = this.f1786g;
            } while (eVar != e.f1793a);
            c(runnable, executor);
        }
        c(runnable, executor);
    }

    @Override // java.util.concurrent.Future
    public final boolean cancel(boolean z) {
        Object obj = this.f1785f;
        if (!(obj == null) && !(obj instanceof g)) {
            return false;
        }
        c cVar = f1781b ? new c(z, new CancellationException("Future.cancel() was called.")) : z ? c.f1788a : c.f1789b;
        while (!f1783d.b(this, obj, cVar)) {
            obj = this.f1785f;
            if (!(obj instanceof g)) {
                return false;
            }
        }
        b(this);
        if (obj instanceof g) {
            Objects.requireNonNull((g) obj);
            throw null;
        }
        return true;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: java.lang.Object */
    /* JADX WARN: Multi-variable type inference failed */
    public final V d(Object obj) {
        if (!(obj instanceof c)) {
            if (!(obj instanceof d)) {
                if (obj == f1784e) {
                    return null;
                }
                return obj;
            }
            throw new ExecutionException(((d) obj).f1792a);
        }
        Throwable th = ((c) obj).f1791d;
        CancellationException cancellationException = new CancellationException("Task was cancelled.");
        cancellationException.initCause(th);
        throw cancellationException;
    }

    public String f() {
        Object obj = this.f1785f;
        if (obj instanceof g) {
            Objects.requireNonNull((g) obj);
            return "setFuture=[null]";
        } else if (this instanceof ScheduledFuture) {
            StringBuilder x = c.b.a.a.a.x("remaining delay=[");
            x.append(((ScheduledFuture) this).getDelay(TimeUnit.MILLISECONDS));
            x.append(" ms]");
            return x.toString();
        } else {
            return null;
        }
    }

    public final void g(i iVar) {
        iVar.f1803b = null;
        while (true) {
            i iVar2 = this.f1787h;
            if (iVar2 == i.f1802a) {
                return;
            }
            i iVar3 = null;
            while (iVar2 != null) {
                i iVar4 = iVar2.f1804c;
                if (iVar2.f1803b != null) {
                    iVar3 = iVar2;
                } else if (iVar3 != null) {
                    iVar3.f1804c = iVar4;
                    if (iVar3.f1803b == null) {
                        break;
                    }
                } else if (!f1783d.c(this, iVar2, iVar4)) {
                    break;
                }
                iVar2 = iVar4;
            }
            return;
        }
    }

    @Override // java.util.concurrent.Future
    public final V get(long j, TimeUnit timeUnit) {
        Locale locale;
        long nanos = timeUnit.toNanos(j);
        if (!Thread.interrupted()) {
            Object obj = this.f1785f;
            boolean z = true;
            if ((obj != null) & (!(obj instanceof g))) {
                return d(obj);
            }
            long nanoTime = nanos > 0 ? System.nanoTime() + nanos : 0L;
            if (nanos >= 1000) {
                i iVar = this.f1787h;
                if (iVar != i.f1802a) {
                    i iVar2 = new i();
                    do {
                        b bVar = f1783d;
                        bVar.d(iVar2, iVar);
                        if (bVar.c(this, iVar, iVar2)) {
                            do {
                                LockSupport.parkNanos(this, nanos);
                                if (!Thread.interrupted()) {
                                    Object obj2 = this.f1785f;
                                    if ((obj2 != null) & (!(obj2 instanceof g))) {
                                        return d(obj2);
                                    }
                                    nanos = nanoTime - System.nanoTime();
                                } else {
                                    g(iVar2);
                                    throw new InterruptedException();
                                }
                            } while (nanos >= 1000);
                            g(iVar2);
                        } else {
                            iVar = this.f1787h;
                        }
                    } while (iVar != i.f1802a);
                    return d(this.f1785f);
                }
                return d(this.f1785f);
            }
            while (nanos > 0) {
                Object obj3 = this.f1785f;
                if ((obj3 != null) & (!(obj3 instanceof g))) {
                    return d(obj3);
                }
                if (!Thread.interrupted()) {
                    nanos = nanoTime - System.nanoTime();
                } else {
                    throw new InterruptedException();
                }
            }
            String aVar = toString();
            String lowerCase = timeUnit.toString().toLowerCase(Locale.ROOT);
            String str = "Waited " + j + " " + timeUnit.toString().toLowerCase(locale);
            if (nanos + 1000 < 0) {
                String q = c.b.a.a.a.q(str, " (plus ");
                long j2 = -nanos;
                long convert = timeUnit.convert(j2, TimeUnit.NANOSECONDS);
                long nanos2 = j2 - timeUnit.toNanos(convert);
                int i2 = (convert > 0L ? 1 : (convert == 0L ? 0 : -1));
                if (i2 != 0 && nanos2 <= 1000) {
                    z = false;
                }
                if (i2 > 0) {
                    String str2 = q + convert + " " + lowerCase;
                    if (z) {
                        str2 = c.b.a.a.a.q(str2, ",");
                    }
                    q = c.b.a.a.a.q(str2, " ");
                }
                if (z) {
                    q = q + nanos2 + " nanoseconds ";
                }
                str = c.b.a.a.a.q(q, "delay)");
            }
            if (isDone()) {
                throw new TimeoutException(c.b.a.a.a.q(str, " but future completed as timeout expired"));
            }
            throw new TimeoutException(c.b.a.a.a.r(str, " for ", aVar));
        }
        throw new InterruptedException();
    }

    public boolean h(V v) {
        if (v == null) {
            v = (V) f1784e;
        }
        if (f1783d.b(this, null, v)) {
            b(this);
            return true;
        }
        return false;
    }

    public boolean i(Throwable th) {
        Objects.requireNonNull(th);
        if (f1783d.b(this, null, new d(th))) {
            b(this);
            return true;
        }
        return false;
    }

    @Override // java.util.concurrent.Future
    public final boolean isCancelled() {
        return this.f1785f instanceof c;
    }

    @Override // java.util.concurrent.Future
    public final boolean isDone() {
        Object obj = this.f1785f;
        return (!(obj instanceof g)) & (obj != null);
    }

    public String toString() {
        String sb;
        StringBuilder sb2 = new StringBuilder();
        sb2.append(super.toString());
        sb2.append("[status=");
        if (this.f1785f instanceof c) {
            sb2.append("CANCELLED");
        } else if (isDone()) {
            a(sb2);
        } else {
            try {
                sb = f();
            } catch (RuntimeException e2) {
                StringBuilder x = c.b.a.a.a.x("Exception thrown from implementation: ");
                x.append(e2.getClass());
                sb = x.toString();
            }
            if (sb != null && !sb.isEmpty()) {
                sb2.append("PENDING, info=[");
                sb2.append(sb);
                sb2.append("]");
            } else if (isDone()) {
                a(sb2);
            } else {
                sb2.append("PENDING");
            }
        }
        sb2.append("]");
        return sb2.toString();
    }

    @Override // java.util.concurrent.Future
    public final V get() {
        Object obj;
        if (!Thread.interrupted()) {
            Object obj2 = this.f1785f;
            if ((obj2 != null) & (!(obj2 instanceof g))) {
                return d(obj2);
            }
            i iVar = this.f1787h;
            if (iVar != i.f1802a) {
                i iVar2 = new i();
                do {
                    b bVar = f1783d;
                    bVar.d(iVar2, iVar);
                    if (bVar.c(this, iVar, iVar2)) {
                        do {
                            LockSupport.park(this);
                            if (!Thread.interrupted()) {
                                obj = this.f1785f;
                            } else {
                                g(iVar2);
                                throw new InterruptedException();
                            }
                        } while (!((obj != null) & (!(obj instanceof g))));
                        return d(obj);
                    }
                    iVar = this.f1787h;
                } while (iVar != i.f1802a);
                return d(this.f1785f);
            }
            return d(this.f1785f);
        }
        throw new InterruptedException();
    }
}