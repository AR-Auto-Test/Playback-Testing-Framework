package com.google.common.flogger.backend;

import c.b.a.a.a;
import com.google.common.flogger.LogContext;
import com.google.common.flogger.MetadataKey;
import com.google.common.flogger.parameter.DateTimeFormat;
import com.google.common.flogger.parameter.Parameter;
import com.google.common.flogger.parameter.ParameterVisitor;
import com.google.common.flogger.parser.MessageBuilder;
import com.google.common.flogger.util.Checks;
import com.google.common.primitives.UnsignedInts;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.Formattable;
import java.util.Formatter;
import java.util.Locale;
import java.util.logging.Level;

/* loaded from: classes.dex */
public final class SimpleMessageFormatter extends MessageBuilder<StringBuilder> implements ParameterVisitor {
    private static final String EXTRA_ARGUMENT_MESSAGE = " [ERROR: UNUSED LOG ARGUMENTS]";
    private static final Locale FORMAT_LOCALE = Locale.ROOT;
    private static final String MISSING_ARGUMENT_MESSAGE = "[ERROR: MISSING LOG ARGUMENT]";
    private final Object[] args;
    private int literalStart;
    private final StringBuilder out;

    /* renamed from: com.google.common.flogger.backend.SimpleMessageFormatter$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$common$flogger$backend$FormatChar;

        static {
            FormatChar.values();
            int[] iArr = new int[10];
            $SwitchMap$com$google$common$flogger$backend$FormatChar = iArr;
            try {
                iArr[FormatChar.STRING.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$common$flogger$backend$FormatChar[FormatChar.DECIMAL.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                $SwitchMap$com$google$common$flogger$backend$FormatChar[FormatChar.BOOLEAN.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                $SwitchMap$com$google$common$flogger$backend$FormatChar[FormatChar.HEX.ordinal()] = 4;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$common$flogger$backend$FormatChar[FormatChar.CHAR.ordinal()] = 5;
            } catch (NoSuchFieldError unused5) {
            }
        }
    }

    /* loaded from: classes.dex */
    public interface SimpleLogHandler {
        void handleFormattedLogMessage(Level level, String str, Throwable th);
    }

    private SimpleMessageFormatter(TemplateContext templateContext, Object[] objArr) {
        super(templateContext);
        this.out = new StringBuilder();
        this.literalStart = 0;
        this.args = (Object[]) Checks.checkNotNull(objArr, "log arguments");
    }

    private static String appendContext(StringBuilder sb, Metadata metadata) {
        KeyValueFormatter keyValueFormatter = new KeyValueFormatter("[CONTEXT ", " ]", sb);
        Tags tags = null;
        for (int i = 0; i < metadata.size(); i++) {
            MetadataKey<?> key = metadata.getKey(i);
            if (!key.equals(LogContext.Key.LOG_CAUSE)) {
                MetadataKey<Tags> metadataKey = LogContext.Key.TAGS;
                if (key.equals(metadataKey)) {
                    tags = metadataKey.cast(metadata.getValue(i));
                } else {
                    key.emit(metadata.getValue(i), keyValueFormatter);
                }
            }
        }
        if (tags != null) {
            tags.emitAll(keyValueFormatter);
        }
        keyValueFormatter.done();
        return sb.toString();
    }

    private static void appendFormatted(StringBuilder sb, Object obj, FormatChar formatChar, FormatOptions formatOptions) {
        int ordinal = formatChar.ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal != 2) {
                    if (ordinal != 3) {
                        if (ordinal == 5 && formatOptions.filter(128, false, false).equals(formatOptions)) {
                            appendHex(sb, (Number) obj, formatOptions);
                            return;
                        }
                    }
                } else if (formatOptions.isDefault()) {
                    if (obj instanceof Character) {
                        sb.append(obj);
                        return;
                    }
                    int intValue = ((Number) obj).intValue();
                    if (Character.isBmpCodePoint(intValue)) {
                        sb.append((char) intValue);
                        return;
                    } else {
                        sb.append(Character.toChars(intValue));
                        return;
                    }
                }
            }
            if (formatOptions.isDefault()) {
                sb.append(obj);
                return;
            }
        } else if (!(obj instanceof Formattable)) {
            if (formatOptions.isDefault()) {
                sb.append(safeToString(obj));
                return;
            }
        } else {
            safeFormatTo((Formattable) obj, sb, formatOptions);
            return;
        }
        String defaultFormatString = formatChar.getDefaultFormatString();
        if (!formatOptions.isDefault()) {
            char c2 = formatChar.getChar();
            if (formatOptions.shouldUpperCase()) {
                c2 = (char) (c2 & 65503);
            }
            StringBuilder appendPrintfOptions = formatOptions.appendPrintfOptions(new StringBuilder("%"));
            appendPrintfOptions.append(c2);
            defaultFormatString = appendPrintfOptions.toString();
        }
        sb.append(String.format(FORMAT_LOCALE, defaultFormatString, obj));
    }

    public static void appendHex(StringBuilder sb, Number number, FormatOptions formatOptions) {
        boolean shouldUpperCase = formatOptions.shouldUpperCase();
        long longValue = number.longValue();
        if (number instanceof Long) {
            appendHex(sb, longValue, shouldUpperCase);
        } else if (number instanceof Integer) {
            appendHex(sb, longValue & UnsignedInts.INT_MASK, shouldUpperCase);
        } else if (number instanceof Byte) {
            appendHex(sb, longValue & 255, shouldUpperCase);
        } else if (number instanceof Short) {
            appendHex(sb, longValue & 65535, shouldUpperCase);
        } else if (number instanceof BigInteger) {
            String bigInteger = ((BigInteger) number).toString(16);
            if (shouldUpperCase) {
                bigInteger = bigInteger.toUpperCase(FORMAT_LOCALE);
            }
            sb.append(bigInteger);
        } else {
            StringBuilder x = a.x("unsupported number type: ");
            x.append(number.getClass());
            throw new RuntimeException(x.toString());
        }
    }

    private static void appendInvalid(StringBuilder sb, Object obj, String str) {
        sb.append("[INVALID: format=");
        sb.append(str);
        sb.append(", type=");
        sb.append(obj.getClass().getCanonicalName());
        sb.append(", value=");
        sb.append(safeToString(obj));
        sb.append("]");
    }

    public static void format(LogData logData, SimpleLogHandler simpleLogHandler) {
        String sb;
        Metadata metadata = logData.getMetadata();
        Throwable th = (Throwable) metadata.findValue(LogContext.Key.LOG_CAUSE);
        boolean z = true;
        if (metadata.size() != 0 && (metadata.size() != 1 || th == null)) {
            z = false;
        }
        if (logData.getTemplateContext() == null) {
            sb = safeToString(logData.getLiteralArgument());
            if (!z) {
                sb = appendContext(new StringBuilder(sb), metadata);
            }
        } else {
            StringBuilder formatMessage = formatMessage(logData);
            sb = z ? formatMessage.toString() : appendContext(formatMessage, metadata);
        }
        simpleLogHandler.handleFormattedLogMessage(logData.getLevel(), sb, th);
    }

    private static StringBuilder formatMessage(LogData logData) {
        SimpleMessageFormatter simpleMessageFormatter = new SimpleMessageFormatter(logData.getTemplateContext(), logData.getArguments());
        StringBuilder build = simpleMessageFormatter.build();
        if (logData.getArguments().length > simpleMessageFormatter.getExpectedArgumentCount()) {
            build.append(EXTRA_ARGUMENT_MESSAGE);
        }
        return build;
    }

    private static String getErrorString(Object obj, RuntimeException runtimeException) {
        String simpleName;
        try {
            simpleName = runtimeException.toString();
        } catch (RuntimeException e2) {
            simpleName = e2.getClass().getSimpleName();
        }
        StringBuilder x = a.x("{");
        x.append(obj.getClass().getName());
        x.append("@");
        x.append(System.identityHashCode(obj));
        x.append(": ");
        x.append(simpleName);
        x.append("}");
        return x.toString();
    }

    private static void safeFormatTo(Formattable formattable, StringBuilder sb, FormatOptions formatOptions) {
        int flags = formatOptions.getFlags() & 162;
        if (flags != 0) {
            flags = ((flags & 32) != 0 ? 1 : 0) | ((flags & 128) != 0 ? 2 : 0) | ((flags & 2) != 0 ? 4 : 0);
        }
        int length = sb.length();
        Formatter formatter = new Formatter(sb, FORMAT_LOCALE);
        try {
            formattable.formatTo(formatter, flags, formatOptions.getWidth(), formatOptions.getPrecision());
        } catch (RuntimeException e2) {
            sb.setLength(length);
            try {
                formatter.out().append(getErrorString(formattable, e2));
            } catch (IOException unused) {
            }
        }
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:9:0x0002 */
    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: java.lang.Object */
    /* JADX DEBUG: Multi-variable search result rejected for r1v1, resolved type: java.lang.Object */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r1v3, types: [java.lang.String] */
    public static String safeToString(Object obj) {
        if (obj != 0) {
            try {
                obj = toString(obj);
                return obj;
            } catch (RuntimeException e2) {
                return getErrorString(obj, e2);
            }
        }
        return "null";
    }

    public static String toString(Object obj) {
        if (!obj.getClass().isArray()) {
            return String.valueOf(obj);
        }
        if (obj instanceof int[]) {
            return Arrays.toString((int[]) obj);
        }
        if (obj instanceof long[]) {
            return Arrays.toString((long[]) obj);
        }
        if (obj instanceof byte[]) {
            return Arrays.toString((byte[]) obj);
        }
        if (obj instanceof char[]) {
            return Arrays.toString((char[]) obj);
        }
        if (obj instanceof short[]) {
            return Arrays.toString((short[]) obj);
        }
        if (obj instanceof float[]) {
            return Arrays.toString((float[]) obj);
        }
        if (obj instanceof double[]) {
            return Arrays.toString((double[]) obj);
        }
        if (obj instanceof boolean[]) {
            return Arrays.toString((boolean[]) obj);
        }
        return Arrays.toString((Object[]) obj);
    }

    @Override // com.google.common.flogger.parser.MessageBuilder
    public void addParameterImpl(int i, int i2, Parameter parameter) {
        getParser().unescape(this.out, getMessage(), this.literalStart, i);
        parameter.accept((ParameterVisitor) this, this.args);
        this.literalStart = i2;
    }

    @Override // com.google.common.flogger.parameter.ParameterVisitor
    public void visit(Object obj, FormatChar formatChar, FormatOptions formatOptions) {
        if (formatChar.getType().canFormat(obj)) {
            appendFormatted(this.out, obj, formatChar, formatOptions);
        } else {
            appendInvalid(this.out, obj, formatChar.getDefaultFormatString());
        }
    }

    @Override // com.google.common.flogger.parameter.ParameterVisitor
    public void visitDateTime(Object obj, DateTimeFormat dateTimeFormat, FormatOptions formatOptions) {
        if (!(obj instanceof Date) && !(obj instanceof Calendar) && !(obj instanceof Long)) {
            StringBuilder sb = this.out;
            StringBuilder x = a.x("%t");
            x.append(dateTimeFormat.getChar());
            appendInvalid(sb, obj, x.toString());
            return;
        }
        StringBuilder appendPrintfOptions = formatOptions.appendPrintfOptions(new StringBuilder("%"));
        appendPrintfOptions.append(formatOptions.shouldUpperCase() ? 'T' : 't');
        appendPrintfOptions.append(dateTimeFormat.getChar());
        this.out.append(String.format(FORMAT_LOCALE, appendPrintfOptions.toString(), obj));
    }

    @Override // com.google.common.flogger.parameter.ParameterVisitor
    public void visitMissing() {
        this.out.append(MISSING_ARGUMENT_MESSAGE);
    }

    @Override // com.google.common.flogger.parameter.ParameterVisitor
    public void visitNull() {
        this.out.append("null");
    }

    @Override // com.google.common.flogger.parameter.ParameterVisitor
    public void visitPreformatted(Object obj, String str) {
        this.out.append(str);
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.common.flogger.parser.MessageBuilder
    public StringBuilder buildImpl() {
        getParser().unescape(this.out, getMessage(), this.literalStart, getMessage().length());
        return this.out;
    }

    private static void appendHex(StringBuilder sb, long j, boolean z) {
        if (j == 0) {
            sb.append(CrashlyticsReportDataCapture.SIGNAL_DEFAULT);
            return;
        }
        String str = z ? "0123456789ABCDEF" : "0123456789abcdef";
        for (int numberOfLeadingZeros = (63 - Long.numberOfLeadingZeros(j)) & (-4); numberOfLeadingZeros >= 0; numberOfLeadingZeros -= 4) {
            sb.append(str.charAt((int) ((j >>> numberOfLeadingZeros) & 15)));
        }
    }
}