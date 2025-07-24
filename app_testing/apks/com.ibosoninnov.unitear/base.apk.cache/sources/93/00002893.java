package com.google.gson.internal.bind.util;

import c.b.a.a.a;
import java.text.ParseException;
import java.text.ParsePosition;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.TimeZone;

/* loaded from: classes2.dex */
public class ISO8601Utils {
    private static final TimeZone TIMEZONE_UTC = TimeZone.getTimeZone("UTC");
    private static final String UTC_ID = "UTC";

    private static boolean checkOffset(String str, int i, char c2) {
        return i < str.length() && str.charAt(i) == c2;
    }

    public static String format(Date date) {
        return format(date, false, TIMEZONE_UTC);
    }

    private static int indexOfNonDigit(String str, int i) {
        while (i < str.length()) {
            char charAt = str.charAt(i);
            if (charAt < '0' || charAt > '9') {
                return i;
            }
            i++;
        }
        return str.length();
    }

    private static void padInt(StringBuilder sb, int i, int i2) {
        String num = Integer.toString(i);
        for (int length = i2 - num.length(); length > 0; length--) {
            sb.append('0');
        }
        sb.append(num);
    }

    /* JADX WARN: Removed duplicated region for block: B:49:0x00d2 A[Catch: IllegalArgumentException -> 0x01bf, NumberFormatException -> 0x01c1, IndexOutOfBoundsException | NumberFormatException | IllegalArgumentException -> 0x01c3, TryCatch #2 {IndexOutOfBoundsException | NumberFormatException | IllegalArgumentException -> 0x01c3, blocks: (B:3:0x0004, B:5:0x0016, B:6:0x0018, B:8:0x0024, B:9:0x0026, B:11:0x0035, B:13:0x003b, B:17:0x0050, B:19:0x0060, B:20:0x0062, B:22:0x006e, B:23:0x0070, B:25:0x0076, B:29:0x0080, B:34:0x0090, B:36:0x0098, B:47:0x00ca, B:49:0x00d2, B:51:0x00d9, B:75:0x0186, B:55:0x00e3, B:56:0x00fe, B:57:0x00ff, B:61:0x011b, B:63:0x0128, B:66:0x0131, B:68:0x0150, B:71:0x015f, B:72:0x0181, B:74:0x0184, B:60:0x010a, B:77:0x01b7, B:78:0x01be, B:40:0x00b0, B:41:0x00b3), top: B:94:0x0004 }] */
    /* JADX WARN: Removed duplicated region for block: B:77:0x01b7 A[Catch: IllegalArgumentException -> 0x01bf, NumberFormatException -> 0x01c1, IndexOutOfBoundsException | NumberFormatException | IllegalArgumentException -> 0x01c3, TryCatch #2 {IndexOutOfBoundsException | NumberFormatException | IllegalArgumentException -> 0x01c3, blocks: (B:3:0x0004, B:5:0x0016, B:6:0x0018, B:8:0x0024, B:9:0x0026, B:11:0x0035, B:13:0x003b, B:17:0x0050, B:19:0x0060, B:20:0x0062, B:22:0x006e, B:23:0x0070, B:25:0x0076, B:29:0x0080, B:34:0x0090, B:36:0x0098, B:47:0x00ca, B:49:0x00d2, B:51:0x00d9, B:75:0x0186, B:55:0x00e3, B:56:0x00fe, B:57:0x00ff, B:61:0x011b, B:63:0x0128, B:66:0x0131, B:68:0x0150, B:71:0x015f, B:72:0x0181, B:74:0x0184, B:60:0x010a, B:77:0x01b7, B:78:0x01be, B:40:0x00b0, B:41:0x00b3), top: B:94:0x0004 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Date parse(String str, ParsePosition parsePosition) {
        String str2;
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        int i6;
        int i7;
        int length;
        TimeZone timeZone;
        char charAt;
        try {
            int index = parsePosition.getIndex();
            int i8 = index + 4;
            int parseInt = parseInt(str, index, i8);
            if (checkOffset(str, i8, '-')) {
                i8++;
            }
            int i9 = i8 + 2;
            int parseInt2 = parseInt(str, i8, i9);
            if (checkOffset(str, i9, '-')) {
                i9++;
            }
            int i10 = i9 + 2;
            int parseInt3 = parseInt(str, i9, i10);
            boolean checkOffset = checkOffset(str, i10, 'T');
            if (!checkOffset && str.length() <= i10) {
                GregorianCalendar gregorianCalendar = new GregorianCalendar(parseInt, parseInt2 - 1, parseInt3);
                parsePosition.setIndex(i10);
                return gregorianCalendar.getTime();
            }
            if (checkOffset) {
                int i11 = i10 + 1;
                int i12 = i11 + 2;
                i6 = parseInt(str, i11, i12);
                if (checkOffset(str, i12, ':')) {
                    i12++;
                }
                int i13 = i12 + 2;
                i7 = parseInt(str, i12, i13);
                if (checkOffset(str, i13, ':')) {
                    i13++;
                }
                if (str.length() > i13 && (charAt = str.charAt(i13)) != 'Z' && charAt != '+' && charAt != '-') {
                    i5 = i13 + 2;
                    i4 = parseInt(str, i13, i5);
                    if (i4 > 59 && i4 < 63) {
                        i4 = 59;
                    }
                    if (checkOffset(str, i5, '.')) {
                        int i14 = i5 + 1;
                        int indexOfNonDigit = indexOfNonDigit(str, i14 + 1);
                        int min = Math.min(indexOfNonDigit, i14 + 3);
                        i3 = parseInt(str, i14, min);
                        int i15 = min - i14;
                        if (i15 == 1) {
                            i3 *= 100;
                        } else if (i15 == 2) {
                            i3 *= 10;
                        }
                        i5 = indexOfNonDigit;
                    } else {
                        i3 = 0;
                    }
                    int i16 = i3;
                    if (str.length() <= i5) {
                        char charAt2 = str.charAt(i5);
                        if (charAt2 == 'Z') {
                            timeZone = TIMEZONE_UTC;
                            length = i5 + 1;
                        } else {
                            if (charAt2 != '+' && charAt2 != '-') {
                                throw new IndexOutOfBoundsException("Invalid time zone indicator '" + charAt2 + "'");
                            }
                            String substring = str.substring(i5);
                            if (substring.length() < 5) {
                                substring = substring + "00";
                            }
                            length = i5 + substring.length();
                            if (!"+0000".equals(substring) && !"+00:00".equals(substring)) {
                                String str3 = "GMT" + substring;
                                TimeZone timeZone2 = TimeZone.getTimeZone(str3);
                                String id = timeZone2.getID();
                                if (!id.equals(str3) && !id.replace(":", "").equals(str3)) {
                                    throw new IndexOutOfBoundsException("Mismatching time zone indicator: " + str3 + " given, resolves to " + timeZone2.getID());
                                }
                                timeZone = timeZone2;
                            }
                            timeZone = TIMEZONE_UTC;
                        }
                        GregorianCalendar gregorianCalendar2 = new GregorianCalendar(timeZone);
                        gregorianCalendar2.setLenient(false);
                        gregorianCalendar2.set(1, parseInt);
                        gregorianCalendar2.set(2, parseInt2 - 1);
                        gregorianCalendar2.set(5, parseInt3);
                        gregorianCalendar2.set(11, i6);
                        gregorianCalendar2.set(12, i7);
                        gregorianCalendar2.set(13, i4);
                        gregorianCalendar2.set(14, i16);
                        parsePosition.setIndex(length);
                        return gregorianCalendar2.getTime();
                    }
                    throw new IllegalArgumentException("No time zone indicator");
                }
                i3 = 0;
                i2 = i7;
                i = i6;
                i10 = i13;
            } else {
                i = 0;
                i2 = 0;
                i3 = 0;
            }
            i4 = 0;
            i5 = i10;
            i6 = i;
            i7 = i2;
            int i162 = i3;
            if (str.length() <= i5) {
            }
        } catch (IndexOutOfBoundsException | NumberFormatException | IllegalArgumentException e2) {
            if (str == null) {
                str2 = null;
            } else {
                str2 = '\"' + str + '\"';
            }
            String message = e2.getMessage();
            if (message == null || message.isEmpty()) {
                StringBuilder x = a.x("(");
                x.append(e2.getClass().getName());
                x.append(")");
                message = x.toString();
            }
            ParseException parseException = new ParseException("Failed to parse date [" + str2 + "]: " + message, parsePosition.getIndex());
            parseException.initCause(e2);
            throw parseException;
        }
    }

    private static int parseInt(String str, int i, int i2) {
        int i3;
        int i4;
        if (i < 0 || i2 > str.length() || i > i2) {
            throw new NumberFormatException(str);
        }
        if (i < i2) {
            i4 = i + 1;
            int digit = Character.digit(str.charAt(i), 10);
            if (digit < 0) {
                StringBuilder x = a.x("Invalid number: ");
                x.append(str.substring(i, i2));
                throw new NumberFormatException(x.toString());
            }
            i3 = -digit;
        } else {
            i3 = 0;
            i4 = i;
        }
        while (i4 < i2) {
            int i5 = i4 + 1;
            int digit2 = Character.digit(str.charAt(i4), 10);
            if (digit2 < 0) {
                StringBuilder x2 = a.x("Invalid number: ");
                x2.append(str.substring(i, i2));
                throw new NumberFormatException(x2.toString());
            }
            i3 = (i3 * 10) - digit2;
            i4 = i5;
        }
        return -i3;
    }

    public static String format(Date date, boolean z) {
        return format(date, z, TIMEZONE_UTC);
    }

    public static String format(Date date, boolean z, TimeZone timeZone) {
        GregorianCalendar gregorianCalendar = new GregorianCalendar(timeZone, Locale.US);
        gregorianCalendar.setTime(date);
        StringBuilder sb = new StringBuilder(19 + (z ? 4 : 0) + (timeZone.getRawOffset() == 0 ? 1 : 6));
        padInt(sb, gregorianCalendar.get(1), 4);
        sb.append('-');
        padInt(sb, gregorianCalendar.get(2) + 1, 2);
        sb.append('-');
        padInt(sb, gregorianCalendar.get(5), 2);
        sb.append('T');
        padInt(sb, gregorianCalendar.get(11), 2);
        sb.append(':');
        padInt(sb, gregorianCalendar.get(12), 2);
        sb.append(':');
        padInt(sb, gregorianCalendar.get(13), 2);
        if (z) {
            sb.append('.');
            padInt(sb, gregorianCalendar.get(14), 3);
        }
        int offset = timeZone.getOffset(gregorianCalendar.getTimeInMillis());
        if (offset != 0) {
            int i = offset / 60000;
            int abs = Math.abs(i / 60);
            int abs2 = Math.abs(i % 60);
            sb.append(offset >= 0 ? '+' : '-');
            padInt(sb, abs, 2);
            sb.append(':');
            padInt(sb, abs2, 2);
        } else {
            sb.append('Z');
        }
        return sb.toString();
    }
}