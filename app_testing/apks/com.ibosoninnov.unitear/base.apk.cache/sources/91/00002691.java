package com.google.firebase.crashlytics.internal.common;

import com.google.android.gms.tasks.Continuation;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.TaskCompletionSource;
import com.google.firebase.crashlytics.internal.common.Utils;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/* loaded from: classes2.dex */
public final class Utils {
    private static final ExecutorService TASK_CONTINUATION_EXECUTOR_SERVICE = ExecutorUtils.buildSingleThreadExecutorService("awaitEvenIfOnMainThread task continuation executor");

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ int f5643a = 0;

    private Utils() {
    }

    public static <T> T awaitEvenIfOnMainThread(Task<T> task) {
        final CountDownLatch countDownLatch = new CountDownLatch(1);
        task.continueWith(TASK_CONTINUATION_EXECUTOR_SERVICE, new Continuation() { // from class: c.d.c.h.e.c.g
            @Override // com.google.android.gms.tasks.Continuation
            public final Object then(Task task2) {
                CountDownLatch countDownLatch2 = countDownLatch;
                int i = Utils.f5643a;
                countDownLatch2.countDown();
                return null;
            }
        });
        countDownLatch.await(4L, TimeUnit.SECONDS);
        if (task.isSuccessful()) {
            return task.getResult();
        }
        if (!task.isCanceled()) {
            if (task.isComplete()) {
                throw new IllegalStateException(task.getException());
            }
            throw new TimeoutException();
        }
        throw new CancellationException("Task is already canceled");
    }

    public static boolean awaitUninterruptibly(CountDownLatch countDownLatch, long j, TimeUnit timeUnit) {
        boolean z = false;
        try {
            long nanos = timeUnit.toNanos(j);
            while (true) {
                try {
                    break;
                } catch (InterruptedException unused) {
                    z = true;
                    nanos = (System.nanoTime() + nanos) - System.nanoTime();
                }
            }
            return countDownLatch.await(nanos, TimeUnit.NANOSECONDS);
        } finally {
            if (z) {
                Thread.currentThread().interrupt();
            }
        }
    }

    public static <T> Task<T> callTask(Executor executor, final Callable<Task<T>> callable) {
        final TaskCompletionSource taskCompletionSource = new TaskCompletionSource();
        executor.execute(new Runnable() { // from class: com.google.firebase.crashlytics.internal.common.Utils.1
            @Override // java.lang.Runnable
            public void run() {
                try {
                    ((Task) callable.call()).continueWith(new Continuation<T, Void>() { // from class: com.google.firebase.crashlytics.internal.common.Utils.1.1
                        /* JADX DEBUG: Method merged with bridge method */
                        @Override // com.google.android.gms.tasks.Continuation
                        public Void then(Task<T> task) {
                            if (task.isSuccessful()) {
                                taskCompletionSource.setResult(task.getResult());
                                return null;
                            }
                            taskCompletionSource.setException(task.getException());
                            return null;
                        }
                    });
                } catch (Exception e2) {
                    taskCompletionSource.setException(e2);
                }
            }
        });
        return taskCompletionSource.getTask();
    }

    public static <T> Task<T> race(Task<T> task, Task<T> task2) {
        final TaskCompletionSource taskCompletionSource = new TaskCompletionSource();
        Continuation<T, TContinuationResult> continuation = new Continuation() { // from class: c.d.c.h.e.c.f
            @Override // com.google.android.gms.tasks.Continuation
            public final Object then(Task task3) {
                TaskCompletionSource taskCompletionSource2 = TaskCompletionSource.this;
                int i = Utils.f5643a;
                if (task3.isSuccessful()) {
                    taskCompletionSource2.trySetResult(task3.getResult());
                    return null;
                }
                Exception exception = task3.getException();
                Objects.requireNonNull(exception);
                taskCompletionSource2.trySetException(exception);
                return null;
            }
        };
        task.continueWith(continuation);
        task2.continueWith(continuation);
        return taskCompletionSource.getTask();
    }

    public static <T> Task<T> race(Executor executor, Task<T> task, Task<T> task2) {
        final TaskCompletionSource taskCompletionSource = new TaskCompletionSource();
        Continuation<T, TContinuationResult> continuation = new Continuation() { // from class: c.d.c.h.e.c.e
            @Override // com.google.android.gms.tasks.Continuation
            public final Object then(Task task3) {
                TaskCompletionSource taskCompletionSource2 = TaskCompletionSource.this;
                int i = Utils.f5643a;
                if (task3.isSuccessful()) {
                    taskCompletionSource2.trySetResult(task3.getResult());
                    return null;
                }
                Exception exception = task3.getException();
                Objects.requireNonNull(exception);
                taskCompletionSource2.trySetException(exception);
                return null;
            }
        };
        task.continueWith(executor, continuation);
        task2.continueWith(executor, continuation);
        return taskCompletionSource.getTask();
    }
}